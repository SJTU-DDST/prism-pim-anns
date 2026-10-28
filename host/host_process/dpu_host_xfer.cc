#include "dpu_host_xfer.hh"


void send_raw_data(dpu::DpuSet& system,
                   const FlatMatrix<DATA_TYPE>& vectors,
                   const JaggedArray<int>& clusters,
                   const JaggedArray<int>& multi_labels)
{
    // 直接从扁平结构获取总数
    const size_t total_vectors = clusters.data.size();
    std::cout << "\n[Prepare] Sending " << total_vectors << " vectors to DPUs ..." << std::endl;
    
    if (total_vectors > (MAX_VECTORS_PER_DPU * NR_DPUS)) {
        throw std::runtime_error("Vectors exceed DPU capacity. Need larger MAX_VECTORS_PER_DPU.");
    }
    
    // 遍历每个类别，发送数据到 DPU
    const uint16_t num_clusters = clusters.size(); 
    for (uint16_t cid = 0; cid < num_clusters; ++cid) {
        if ((cid + 1) % 512 == 0) {
            std::cout << "[Prepare progress] Send and store " << (cid + 1) << "/" 
                      << num_clusters << " clusters ..." << std::endl;
        }
        // 获取一个cluster的成员ID视图(span)，无内存拷贝
        auto vids_in_cluster = clusters[cid];

        // 调用重构后的辅助函数，将视图传递下去
        dispatch_cluster_in_hash(system, vectors, vids_in_cluster, multi_labels, cid);
    }
    std::cout << "[Prepare done] Send and store raw data done." << std::endl;
}


void dispatch_cluster_in_block(dpu::DpuSet& system,
                               const FlatMatrix<DATA_TYPE>& vectors,
                               std::span<const int> vids_in_cluster, 
                               const JaggedArray<int>& multi_labels, 
                               const int curr_cid)
{
    int wait_to_send_num = vids_in_cluster.size();
    int dpu_vector_index = 0;

    uint8_t is_new = 1; 
    while (wait_to_send_num > 0) {
        int curr_send_num = std::min(wait_to_send_num, DATA_BATCH_SIZE * NR_DPUS);
        int base_num_per_dpu = curr_send_num / NR_DPUS;
        int extra = curr_send_num % NR_DPUS;

        std::vector<std::vector<uint8_t>> dpu_buffers(NR_DPUS, std::vector<uint8_t>(DATA_PUSH_SIZE, 0));

        for (int i = 0; i < NR_DPUS; ++i) {
            // 计算当前 batch 给当前 DPU 分配的 vector 数量
            uint16_t assigned_num = base_num_per_dpu + (i < extra ? 1 : 0);
            DataPackage* package = reinterpret_cast<DataPackage*>(dpu_buffers[i].data());
            
            // 填写数据包头部
            package->header.option_type = 1;  // 1 表示存储数据，2 表示查询
            package->header.is_new = is_new;
            package->header.num_vectors = assigned_num;
            package->header.data_cluster = curr_cid;

            // 如果一个DPU被分配到0个向量，则跳过后续操作
            // std::cout << "assigned_num: " << assigned_num << std::endl;
            // if (assigned_num == 0) continue;

            // 填充 vector_id 和 vector 本身
            for (uint16_t j = 0; j < assigned_num; j++) {
                int vector_id = vids_in_cluster[dpu_vector_index];
                auto vec_view = vectors[vector_id];  // 从 FlatMatrix 获取向量视图

                // std::cout << "cluster_id: " << curr_cid << ", vector_id: " << vector_id
                //           << ", ccount: " << multi_labels[vector_id].size() << std::endl;  // 检查使用

                package->data[j].vector_id = vector_id;

                #if (MIN_REPLICA != 1) 
                    // 从 JaggedArray 获取多标签视图
                    auto labels_view = multi_labels[vector_id];
                    uint16_t k = (labels_view[0] == curr_cid) ? 1 : 0;
                    package->data[j].cross_cid = (uint16_t)labels_view[k];
                #endif

                // memcpy 的源地址现在是视图的 data() 指针
                std::memcpy(package->data[j].vector, vec_view.data(), sizeof(DATA_TYPE) * VECTOR_DIM);
                dpu_vector_index++;
            }
        }
        system.copy("recv_buffer", 0, dpu_buffers, DATA_PUSH_SIZE);
        system.exec();  // 触发 DPU 执行数据接收

        wait_to_send_num -= curr_send_num;
        is_new = 0;
    }
    // assert(dpu_vector_index == vids_in_cluster.size()); // 确保所有向量都发给了 DPU
}


void dispatch_cluster_in_hash(dpu::DpuSet& system,
                              const FlatMatrix<DATA_TYPE>& vectors,
                              std::span<const int> vids_in_cluster, 
                              const JaggedArray<int>& multi_labels, 
                              const int curr_cid)
{
    int wait_to_send_num = vids_in_cluster.size(); 
    int dpu_vector_index = 0;

    uint8_t is_new = 1;  // 标志是否为当前 cluster 发给 DPU 的第一个 batch（如果是，DPU 中需新开一个 offset）
    while (wait_to_send_num > 0) {
        int curr_send_num = std::min(wait_to_send_num, DATA_BATCH_SIZE * NR_DPUS);
    
        std::vector<std::vector<uint8_t>> dpu_buffers(NR_DPUS, std::vector<uint8_t>(DATA_PUSH_SIZE, 0));
        std::vector<int> dpu_counts(NR_DPUS, 0);  // 每个 DPU 当前已经填了几个数据
        std::vector<DataPackage*> packages(NR_DPUS);

        // 初始化 header
        for (int i = 0; i < NR_DPUS; ++i) {
            packages[i] = reinterpret_cast<DataPackage*>(dpu_buffers[i].data());
            packages[i]->header.option_type = 1;
            packages[i]->header.is_new = is_new;
            packages[i]->header.data_cluster = curr_cid;
        }
        
        for (int i = 0; i < curr_send_num; ++i) {
            int vector_id = vids_in_cluster[dpu_vector_index];
            auto vec_view = vectors[vector_id];  // 从 FlatMatrix 获取向量视图

            // 方法一：用 mod NR_DPUS 分配，效果跟 hash 差不多，搭配“用方差作为hint的提前退出计算方案”效果会更好一些
            int dpu_id = i % NR_DPUS;

            // 方法二：用 hash 分配，注意 hash 做不到完全平均，所以 DATA_PUSH_SIZE 要更大一点，否则会出现内存覆写
            // int dpu_id = hash_function(vector_id + 0x2946534) % NR_DPUS; // 哈希分配 DPU

            int offset = dpu_counts[dpu_id];
            packages[dpu_id]->data[offset].vector_id = vector_id;
            
            #if (MIN_REPLICA != 1)
                // 从 JaggedArray 获取多标签视图
                auto labels_view = multi_labels[vector_id];
                uint16_t k = (labels_view[0] == curr_cid) ? 1 : 0;
                packages[dpu_id]->data[offset].cross_cid = (uint16_t)labels_view[k];
            #endif

            // memcpy 的源地址现在是视图的 data() 指针
            std::memcpy(packages[dpu_id]->data[offset].vector, vec_view.data(), sizeof(DATA_TYPE) * VECTOR_DIM);
    
            dpu_counts[dpu_id]++;
            dpu_vector_index++;
        }
    
        for (int i = 0; i < NR_DPUS; ++i) {
            packages[i]->header.num_vectors = dpu_counts[i];  // 更新每个 DPU 的 num_vectors
        }
        system.copy("recv_buffer", 0, dpu_buffers, DATA_PUSH_SIZE);
        system.exec();

        wait_to_send_num -= curr_send_num;
        is_new = 0;
    }
    // assert(dpu_vector_index == vids_in_cluster.size()); // 确保所有向量都发给了 DPU
}


void broadcast_query_batch(dpu::DpuSet& system, 
                           const FlatMatrix<DATA_TYPE>& query_vectors,
                           const FlatMatrix<int>& query_cids, 
                           const std::vector<uint8_t>& split_indices, 
                           const int qc_num,
                           const FlatMatrix<uint8_t>& var_score_list, 
                           const int query_offset)
{
    std::vector<uint8_t> dpu_buffers(QUERY_PUSH_SIZE, 0);  // 创建发送缓冲区
    const uint16_t batch_size = query_vectors.size();

    QueryPackage* package = reinterpret_cast<QueryPackage*>(dpu_buffers.data());
    package->header.option_type = 2;  // 查询
    package->header.num_vectors = batch_size;
    // package->header.data_cluster = 0; // 对查询无意义
    package->header.is_new = (query_offset == 0) ? 1 : 0; // 标注是否为第一批查询，用于重置 DPU 计时器

    // 填充每个query的多个cluster信息
    for (uint16_t j = 0; j < batch_size; ++j) {
        // 一次性获取当前查询所需的所有数据行的视图/指针
        auto qvec_view = query_vectors[j];
        const int* qc_ids_row = query_cids[j].data(); // 从 std::span 中获取其裸指针，需使用 .data() 成员函数
        const uint8_t* var_scores_row = var_score_list[j].data(); // 指向第j个查询的所有排序后的segment信息

        // 填充元数据
        package->data[j].query_id = query_offset + (int)j; 
        package->data[j].split_index = split_indices[j];
        package->data[j].qc_num = qc_num;
        
        // 填充每个查询需要访问的 cluster_id 和对应的 segment 检查顺序
        for (int cidx = 0; cidx < qc_num; ++cidx) {
            package->data[j].cluster_ids[cidx] = qc_ids_row[cidx];  // 通过行指针访问数据
            
            // 计算 memcpy 的源地址：当前查询的 var_score 行的起始地址 + 第 cidx 个簇的偏移量
            const uint8_t* source_ptr = var_scores_row + cidx * NUM_CHECKPOINTS;
            std::memcpy(package->data[j].check_dims[cidx], source_ptr, NUM_CHECKPOINTS);
        }
        
        std::memcpy(package->data[j].vector, qvec_view.data(), sizeof(DATA_TYPE) * VECTOR_DIM);
        
        // 可选调试输出
        // std::cout << "Query " << j << ":";
        // for (int i = 0; i < VECTOR_DIM; i++) {
        //     std::cout << static_cast<int>(package->data[j].vector[i]) << " ";
        // }
        // std::cout << std::endl;
        // for (int cidx = 0; cidx < MAX_QUERY_CLUSTER; ++cidx) {
        //     for (int k = 0; k < NUM_CHECKPOINTS; ++k) {
        //         std::cout << "Query " << j << ", cidx-" << cidx
        //                   << ": dim = " << (int)package->data[j].check_dims[cidx][k] << std::endl;
        //         assert((int)package->data[j].check_dims[cidx][k] < VECTOR_DIM);
        //     }
        // }
    }

    system.copy("recv_buffer", 0, dpu_buffers, QUERY_PUSH_SIZE);
    // std::cout << "Sending query data to DPU, query offset: " << query_offset << std::endl;
    system.exec();
}


void pull_query_result(dpu::DpuSet& system, 
                       FlatMatrix<std::pair<uint32_t, int>>& results, 
                       DpuStats& dpu_stats_info, 
                       const int query_offset, const uint16_t batch_size) 
{
    static std::vector<std::vector<uint8_t>> dpu_buffers(NR_DPUS, std::vector<uint8_t>(PULL_RESULT_SIZE, 0));
    system.copy(dpu_buffers, PULL_RESULT_SIZE, "ret_buffer", 0);
    // std::cout << "Copy results from dpu success." << std::endl;

    dpu_stats_info.start_new_batch();
    for (int i = 0; i < NR_DPUS; ++i) {
        // printf("DPU id:%i\n", i);
        const StatsInfo* batch_stats = reinterpret_cast<const StatsInfo*>(dpu_buffers[i].data());
        const OneQueryResult* query_results = reinterpret_cast<const OneQueryResult*>(dpu_buffers[i].data() + sizeof(StatsInfo));

        // 遍历这个DPU返回的属于当前批次的所有查询结果
        for (uint16_t j = 0; j < batch_size; ++j) {
            // 计算当前查询在批次中的局部行号 (0 to batch_size-1)
            int batch_query_id = query_results[j].query_id - query_offset;
            int row_start_index = batch_query_id * results.dimension;  // 计算当前行的起始索引

            // 遍历当前查询从这个DPU拿到的所有邻居
            for (int k = 0; k < MAX_RETURNED_PER_DPU; ++k) {
                // 计算结果应该写入的列号
                int col_index = i * MAX_RETURNED_PER_DPU + k;

                // 计算在一维 vector<T> data 中的最终目标索引，对展平的 data 向量进行写操作
                int dest_index = row_start_index + col_index;

                results.data[dest_index].first = query_results[j].distance[k];
                results.data[dest_index].second = query_results[j].neighbor_ids[k];

                // 检查索引是否越界
                // if (dest_index < results.data.size()) {
                //     results.data[dest_index].first = query_results[j].distance[k];
                //     results.data[dest_index].second = query_results[j].neighbor_ids[k];
                // } else {
                //     // 如果发生越界，打印错误信息进行调试
                //     printf("batch_size:%hd, offset:%d, batch_query_id:%d, dimension:%ld\n", batch_size, query_offset, batch_query_id, results.dimension);
                //     printf("Error Index:%d, bounds:%ld. col_index:%d, row_start_index:%d\n", dest_index, results.data.size(), col_index, row_start_index);
                //     std::cout << "query_results[" << j << "].distance[" << k << "] " << query_results[j].distance[k]
                //               << ", query_results[" << j << "].neighbor_ids[" << k << "] " << query_results[j].neighbor_ids[k] << std::endl;
                // }
            }
        }
        dpu_stats_info.add(batch_stats, i);  // 聚合统计信息
    }
    dpu_stats_info.finalize_batch();
    // std::cout << "Batch query result successfully retrieved from DPU." << std::endl;
}


void pull_results_and_extract_topk(dpu::DpuSet& system, 
                                   FlatMatrix<int>& topk_results, 
                                   DpuStats& dpu_stats_info, 
                                   const int query_offset, 
                                   const uint16_t batch_size)
{
    std::vector<std::vector<uint8_t>> dpu_buffers(NR_DPUS, std::vector<uint8_t>(PULL_RESULT_SIZE, 0));
    system.copy(dpu_buffers, PULL_RESULT_SIZE, "ret_buffer", 0);
    // std::cout << "Copy results from dpu success." << std::endl;

    dpu_stats_info.start_new_batch();

    // 每个 batch_query 的 TopK 聚合
    for (uint16_t j = 0; j < batch_size; ++j) {
        // Top-K 提取和去重逻辑
        std::priority_queue<std::pair<uint32_t, int>> pq;
        std::unordered_set<int> seen_ids;
        seen_ids.reserve(NR_DPUS * MAX_RETURNED_PER_DPU);

        for (int i = 0; i < NR_DPUS; ++i) {
            OneQueryResult* query_results = reinterpret_cast<OneQueryResult*>(dpu_buffers[i].data() + sizeof(StatsInfo));
            const OneQueryResult& res = query_results[j];
            
            for (int k = 0; k < MAX_RETURNED_PER_DPU; ++k) {
                uint32_t dist = res.distance[k];
                int vid = res.neighbor_ids[k];
                
                if (dist == UINT32_MAX) continue;
                if (seen_ids.count(vid)) continue;
                seen_ids.insert(vid);

                if (pq.size() < TOP_K) {
                    pq.emplace(dist, vid);
                } else if (dist < pq.top().first) {
                    pq.pop();
                    pq.emplace(dist, vid);
                }
            }
        }

        // 将最终结果写入扁平化的 topk_results 矩阵
        int global_query_id = query_offset + j;
        // 获取指向目标行的指针
        int* topk_row = topk_results[global_query_id].data();

        // 从优先队列中取出结果，并填充到目标行
        // topk_results[global_query_id] 现在已经升序（最小距离ID在前）
        for (int t = TOP_K - 1; t >= 0; --t) {
            topk_row[t] = pq.top().second;
            pq.pop();
        }
    }

    // 聚合每个 DPU 的统计信息
    for (int i = 0; i < NR_DPUS; ++i) {
        const StatsInfo* batch_stats = reinterpret_cast<StatsInfo*>(dpu_buffers[i].data());
        dpu_stats_info.add(batch_stats, i);
    }
    dpu_stats_info.finalize_batch();
    // std::cout << "Batch query result successfully retrieved from DPU." << std::endl;
}


void simple_pull_results(dpu::DpuSet& system, std::vector<std::vector<uint8_t>>& pull_buffers) 
{
    system.copy(pull_buffers, PULL_RESULT_SIZE, "ret_buffer", 0);
}


void simple_broadcast_query(dpu::DpuSet& system, DpuStats& dpu_stats_info, 
                            const std::chrono::steady_clock::time_point& global_start, 
                            std::vector<uint8_t>& broadcast_buffer) 
{
    system.copy("recv_buffer", 0, broadcast_buffer, QUERY_PUSH_SIZE);
    system.exec();
    // system.log(std::cout);
}


void pull_batch_time_stamp(dpu::DpuSet& system, std::vector<std::vector<uint64_t>>& time_buffers) 
{
    // 从DPU的 "dpu_time_stats" 符号处拷贝数据
    system.copy(time_buffers, MAX_DPU_TIME_STAMP * sizeof(uint64_t), "dpu_time_stats", 0);
}
