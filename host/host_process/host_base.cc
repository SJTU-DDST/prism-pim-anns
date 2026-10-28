#include "host_base.hh"
#include <fstream>


void compute_query_segment_scores(const FlatMatrix<DATA_TYPE>& query_vectors, 
                                 const FlatMatrix<float>& centroids, 
                                 const FlatMatrix<int>& query_clusters, 
                                 const FlatMatrix<float>& cluster_var_list, 
                                 FlatMatrix<uint8_t>& var_score_list, 
                                 const int qc_num,
                                 float alpha, float beta)
{
    const uint16_t batch_size = query_vectors.size();
    
    for (uint16_t i = 0; i < batch_size; ++i) {
        // 在循环外层一次性获取当前查询向量的视图
        auto qvec = query_vectors[i].data();
        const int* qc_row = query_clusters[i].data();
        
        // 获取指向输出矩阵中当前查询对应行的指针
        // var_score_list 的维度是 (NUM_CLUSTER * NUM_CHECKPOINTS)
        uint8_t* var_score_row = var_score_list[i].data();

        for (int j = 0; j < qc_num; ++j) {
            int cidx = qc_row[j];
            // assert(cidx >= 0 && cidx < centroids.size() && "Cluster ID is out of bounds for centroids matrix!");
            // assert(cidx >= 0 && cidx < cluster_var_list.size() && "Cluster ID is out of bounds for variance matrix!");

            // 一次性获取质心和方差的视图
            auto centroid_vec = centroids[cidx];
            auto cluster_var_vec = cluster_var_list[cidx];

            // 统计每个 segment 的总分（目前计算总和）
            std::vector<float> segment_scores(NUM_CHECKPOINTS, 0.0f);

            // 这里的方差是按照每个维度算的，实际上可以按照段来保存
            for (int d = 0; d < VECTOR_DIM; ++d) {
                float query_val = static_cast<float>(qvec[d]);
                float delta = query_val - centroid_vec[d];
                float delta_sq = delta * delta;
                float score = alpha * cluster_var_vec[d] + beta * delta_sq;

                int segment_id = d / BUCKET_SIZE;
                segment_scores[segment_id] += score;
            }

            // 将 segment 排序：<score, segment_id>
            std::vector<std::pair<float, uint8_t>> sorted_segments;
            sorted_segments.reserve(NUM_CHECKPOINTS); // 预分配空间
            for (int b = 0; b < NUM_CHECKPOINTS; ++b) {
                sorted_segments.emplace_back(segment_scores[b], b);
            }

            std::sort(sorted_segments.begin(), sorted_segments.end(),
                      [](const auto& a, const auto& b) {
                          return a.first > b.first;  // 大分数在前
                      });

            // 保存排序后的 segment 起始维度（作为 DPU 查询顺序）
            for (int k = 0; k < NUM_CHECKPOINTS; ++k) {
                int segment_id = sorted_segments[k].second;
                // assert((segment_id >= 0) && (segment_id < VECTOR_DIM));
                // 索引 = j * 单个二维块的大小 + k
                var_score_row[j * NUM_CHECKPOINTS + k] = static_cast<uint8_t>(segment_id * BUCKET_SIZE);

                // 可选调试
                // std::cout << "Query " << i << ", Cluster " << j
                //           << ", Segment[" << k << "] = " << (int)var_score_list[i][j * NUM_CHECKPOINTS + k]
                //           << " (score = " << sorted_segments[k].first << ")\n";
            }
        }
    }
}


// 私有实现
void find_query_clusters_impl(const FlatMatrix<DATA_TYPE>& query_vectors, 
                              const FlatMatrix<float>& centroids, 
                              FlatMatrix<int>& query_clusters, 
                              std::vector<uint8_t>* split_indices, 
                              const int qc_num)
{
    const uint16_t batch_size = query_vectors.size();
    const int dim = query_vectors.dim();

    // 临时缓冲，用于存每个 query 到所有 cluster 的距离
    std::vector<std::pair<float, int>> dist_buffer(NUM_CLUSTER);
    // 结果保存，记录 query vector 到最近的若干个质心的距离
    std::vector<float> dist_to_centroid(qc_num);

    for (uint16_t i = 0; i < batch_size; ++i) {
        // qvec 现在是一个视图(span)或指针
        const auto qvec = query_vectors[i].data();

        // 使用 SIMD 计算当前 query 到所有质心距离
        for (uint16_t j = 0; j < NUM_CLUSTER; ++j) {
            // centroid 同样是视图
            float dist_sq = l2_distance_squared_avx(qvec, centroids[j].data(), dim);
            dist_buffer[j] = {dist_sq, j};
        }

        // 对距离平方进行排序，获取本轮 nprobe 个最近的质心
        std::partial_sort(dist_buffer.begin(), 
                          dist_buffer.begin() + qc_num,
                          dist_buffer.end());

        // 获取指向当前查询结果行的指针
        int* query_cluster_row = query_clusters[i].data();

        for (int k = 0; k < qc_num; ++k) {
            float curr_dist = dist_buffer[k].first;
            int cluster_id = dist_buffer[k].second;

            // 通过指针写入扁平化内存
            query_cluster_row[k] = cluster_id;
            dist_to_centroid[k] = curr_dist;
        }

        // split index 根据 weighted_dist 分割
        if (split_indices) {
            // 计算加权距离，作为分割距离
            float weighted_dist = weighted_dist_to_cluster(dist_to_centroid, qc_num);
            // 使用 std::lower_bound 找到第一个大于或等于 weighted_dist 的位置
            auto it = std::lower_bound(dist_to_centroid.begin(),
                                       dist_to_centroid.begin() + qc_num,
                                       weighted_dist);
            
            uint32_t split_index = std::distance(dist_to_centroid.begin(), it);
            // uint32_t split_index = 0; // 测试所有 cluster 都按照交集方案查询

            (*split_indices)[i] = static_cast<uint8_t>(split_index); 
            // 把后面的交集部分的 cluster id 排序，以便 DPU 中使用双指针快速筛选和匹配
            std::sort(query_cluster_row + split_index, query_cluster_row + qc_num);
        }
    }
    // std::cout << std::endl;
}


// 公开接口1：本批次共用 qc_num，采用交集查询法
void find_query_clusters(const FlatMatrix<DATA_TYPE>& query_vectors, 
                         const FlatMatrix<float>& centroids, 
                         FlatMatrix<int>& query_clusters, 
                         std::vector<uint8_t>& split_indices, 
                         const int qc_num)
{
    find_query_clusters_impl(query_vectors, centroids, query_clusters, 
                             &split_indices, qc_num);
}

// 公开接口2：本批次共用 qc_num，不计算交集分界
void find_query_clusters(const FlatMatrix<DATA_TYPE>& query_vectors, 
                         const FlatMatrix<float>& centroids, 
                         FlatMatrix<int>& query_clusters,
                         const int qc_num)
{
    find_query_clusters_impl(query_vectors, centroids, query_clusters,
                             nullptr, qc_num);
}


void process_results_and_rerank(DpuStats& dpu_stats_info, 
                                FlatMatrix<int>& topk_results, 
                                std::shared_ptr<Batch> completed_batch)
{
    // 1. 引用 Batch 内的数据
    std::vector<std::vector<uint8_t>>& dpu_buffers = completed_batch->pull_buffers;
    const uint16_t batch_size = completed_batch->size;
    const int query_offset = completed_batch->offset;

    // 2. 聚合每个 DPU 的统计信息
    dpu_stats_info.start_new_batch();
    for (int i = 0; i < NR_DPUS; ++i) {
        const StatsInfo* batch_stats = reinterpret_cast<const StatsInfo*>(dpu_buffers[i].data());
        dpu_stats_info.add(batch_stats, i);
    }
    dpu_stats_info.finalize_batch();

    // 使用 vector 作为临时存储，后面使用 std::nth_element 排序，比优先队列效率更高
    std::vector<std::pair<uint32_t, int>> candidates;
    candidates.reserve(NR_DPUS * MAX_RETURNED_PER_DPU);
    std::unordered_set<int> seen_ids;
    seen_ids.reserve(NR_DPUS * MAX_RETURNED_PER_DPU);

    for (uint16_t j = 0; j < batch_size; ++j) {
        candidates.clear();
        seen_ids.clear();

        // 1. 收集所有不重复的候选者
        for (int i = 0; i < NR_DPUS; ++i) {
            OneQueryResult* query_results = reinterpret_cast<OneQueryResult*>(dpu_buffers[i].data() + sizeof(StatsInfo));
            const OneQueryResult& res = query_results[j];

            for (int k = 0; k < MAX_RETURNED_PER_DPU; ++k) {
                uint32_t dist = res.distance[k];
                int vid = res.neighbor_ids[k];
                
                // if (dist == UINT32_MAX) continue;  // 通常不会返回 UINT32_MAX
                
                // 使用 insert 的返回值来判断是否是新元素
                if (seen_ids.insert(vid).second) {
                    candidates.emplace_back(dist, vid);
                }
            }
        }

        // 2. 使用 nth_element 找到 top-K
        std::nth_element(candidates.begin(), 
                         candidates.begin() + TOP_K -1, 
                         candidates.end());
        
        // 3. 写入最终结果
        int* topk_row = topk_results[query_offset + j].data();
        for(int t = 0; t < TOP_K; ++t) {
            topk_row[t] = candidates[t].second;
        }
    }
}


void prepare_broadcast_buffer(std::shared_ptr<Batch> batch_to_prepare)
{
    // 获取对Batch内部广播缓冲区的引用，我们将在其上操作
    std::vector<uint8_t>& broadcast_buffer = batch_to_prepare->broadcast_buffer;
    int batch_size = batch_to_prepare->size;
    const int qc_num = batch_to_prepare->qc_num;

    // 断言检查，确保缓冲区已经被正确分配了大小
    // assert(broadcast_buffer.size() == QUERY_PUSH_SIZE && "Broadcast buffer has incorrect size!");

    // 将缓冲区内存解释为 QueryPackage 结构体指针，方便填充
    QueryPackage* package = reinterpret_cast<QueryPackage*>(broadcast_buffer.data());
    
    // --- 1. 填充 header 信息 ---
    package->header.option_type = 2; // 选项类型: 查询
    package->header.num_vectors = batch_size;
    // package->header.data_cluster = 0; // 对查询无意义
    package->header.is_new = (batch_to_prepare->offset == 0) ? 1 : 0; // 标注是否为第一批查询，用于重置 DPU 计时器

    // --- 2. 遍历Batch中的每个查询，填充详细数据 ---
    for (uint16_t j = 0; j < batch_size; ++j) {
        // 从 batch_to_prepare 对象中获取所有需要的数据源的引用或指针
        auto qvec_view = batch_to_prepare->vectors[j];
        const int* qc_ids_row = batch_to_prepare->labels[j].data();
        const uint8_t* var_scores_row = batch_to_prepare->var_score_list[j].data();
        
        // 填充元数据
        package->data[j].query_id = batch_to_prepare->offset + j; 
        package->data[j].split_index = batch_to_prepare->split_indices[j];
        package->data[j].qc_num = qc_num;
        
        // 填充每个查询需要访问的 cluster_id 和对应的 segment 检查顺序
        for (int cidx = 0; cidx < qc_num; ++cidx) {
            package->data[j].cluster_ids[cidx] = qc_ids_row[cidx];
            
            const uint8_t* source_ptr = var_scores_row + cidx * NUM_CHECKPOINTS;
            std::memcpy(package->data[j].check_dims[cidx], source_ptr, NUM_CHECKPOINTS);
        }
        
        // 拷贝查询向量本身
        std::memcpy(package->data[j].vector, qvec_view.data(), sizeof(DATA_TYPE) * VECTOR_DIM);
    }
}


void rerank_and_store_topk(const FlatMatrix<std::pair<uint32_t, int>>& batch_results,
                           FlatMatrix<int>& global_topk_results,
                           const uint16_t batch_size,
                           const int query_offset)
{
    // 在循环外部创建数据结构，避免重复内存分配
    std::priority_queue<std::pair<uint32_t, int>> pq;
    std::unordered_set<int> seen_ids;
    seen_ids.reserve(batch_results.dimension);  // 预分配空间

    // 对批次中的每一个查询进行处理
    for (uint16_t i = 0; i < batch_size; ++i) {
        seen_ids.clear();  // 清空 seen_ids

        // priority_queue需要用循环pop的方式清空。其实pq上一轮结束后就是空的，不需要清空
        while (!pq.empty()) {
            pq.pop();
        }

        // 获取当前查询的原始结果行视图
        auto raw_results_row = batch_results[i];

        // 使用最大堆来高效地找到距离最小的 TopK 个元素
        std::priority_queue<std::pair<uint32_t, int>> pq; // 默认是最大堆

        // 使用哈希集合来对邻居ID进行去重
        std::unordered_set<int> seen_ids;
        seen_ids.reserve(batch_results.dimension);

        // 遍历当前查询从所有DPU返回的全部结果
        for (int j = 0; j < batch_results.dimension; ++j) {
            const auto& result = raw_results_row[j];
            uint32_t dist = result.first;
            int vid = result.second;

            if (dist == UINT32_MAX) continue;  // 跳过无效结果

            // 使用insert的返回值来判断是否是新元素
            // unordered_set::insert 返回一个 std::pair<iterator, bool>，插入成功为 true
            // 新结果的距离比堆顶（当前最大）的还小，则加入堆
            if (seen_ids.insert(vid).second) {
                if (pq.size() < TOP_K) {
                    pq.emplace(dist, vid);
                } else if (dist < pq.top().first) {
                    pq.pop();
                    pq.emplace(dist, vid);
                }
            }
        }

        // --- 将堆中的 TopK 结果写入全局矩阵 ---
        // 计算当前查询在全局结果矩阵中的行号
        int global_row_idx = query_offset + i;
        
        // 获取全局结果矩阵中对应行的视图
        auto output_row = global_topk_results[global_row_idx];

        // 从堆中提取结果。注意：结果是按距离从大到小排列的。在后续计算准确率的时候，需要按照id重新排序
        int k = pq.size() - 1;
        while (!pq.empty()) {
            output_row[k--] = pq.top().second; // 将ID存入最终结果
            pq.pop();
        }
    }
}


void rerank_and_calc_accuracy(const FlatMatrix<int32_t>& ground_truth, 
                              const FlatMatrix<std::pair<uint32_t, int>>& results, 
                              const uint16_t batch_size, double& global_accuracy)
{
    double batch_accuracy = 0.0;

    for (uint16_t i = 0; i < batch_size; ++i) {
        // 构建最大堆（按距离），提取 TopK
        std::priority_queue<std::pair<uint32_t, int>> pq;

        std::unordered_set<int> seen_ids; // 去除重复元素
        seen_ids.reserve(MAX_RETURNED_PER_DPU * NR_DPUS);

        // results[i] 返回一个 span 视图，可以直接用于范围for循环
        for (const auto& result : results[i]) {
            uint32_t dist = result.first;
            int vid = result.second;

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

        // 提取 TopK ID 
        std::vector<int> topk_ids;
        topk_ids.reserve(TOP_K);
        while (!pq.empty()) {
            topk_ids.push_back(pq.top().second);
            pq.pop();
        }

        // 将按 ID 升序排序，为了后面进行比较。
        std::sort(topk_ids.begin(), topk_ids.end());

        // 从 FlatMatrix 中创建一个可供排序的 ground_truth 副本
        auto gt_view = ground_truth[i];
        std::vector<int> gt_ids(gt_view.begin(), gt_view.end()); // 从视图构造一个 vector
        std::sort(gt_ids.begin(), gt_ids.end());

        // 双指针计算交集，统计准确率
        int match_count = 0;
        size_t gi = 0, ti = 0;
        while (gi < gt_ids.size() && ti < topk_ids.size()) {
            if (gt_ids[gi] == topk_ids[ti]) {
                ++match_count;
                ++gi;
                ++ti;
            } else if (gt_ids[gi] < topk_ids[ti]) {
                // std::cout << "Query " << i << " missed truth IDs: " << gt_ids[gi] << "\n";
                ++gi;
            } else {
                ++ti;
            }
        }

        // 调试用，输出剩余未匹配的 ground truth
        // while (gi < gt_ids.size()) {
        //     std::cout << "Query " << i << " missed truth IDs: " << gt_ids[gi] << "\n";
        //     ++gi;
        // }

        double acc = gt_ids.empty() ? 0.0 : static_cast<double>(match_count) / gt_ids.size();
        batch_accuracy += acc;
        global_accuracy += acc;
        // std::cout << "Query: " << i << ", Match count: " << match_count << std::endl;
    }
    std::cout << "Batch accuracy: " << (batch_size > 0 ? batch_accuracy / batch_size : 0.0) << std::endl;
}


void calculate_accuracy(const FlatMatrix<int32_t>& ground_truth,
                        FlatMatrix<int>& topk_results,
                        const int start_idx, const uint16_t calc_size, 
                        const bool is_total)
{
    double accuracy = 0.0;
    // assert ((start_idx + calc_size) <= topk_results.size() && 
    //         (start_idx + calc_size) <= ground_truth.size());  // 确保索引有效

    for (uint32_t i = start_idx; i < (start_idx + calc_size); ++i) {
        // topk_results[i] 返回一个 span 视图，它有 begin() 和 end() 方法
        auto topk_row_view = topk_results[i];
        std::sort(topk_row_view.begin(), topk_row_view.end());

        // 从 FlatMatrix 的视图创建一个可供排序的副本
        auto gt_view = ground_truth[i];
        std::vector<int> gt_ids(gt_view.begin(), gt_view.end());
        std::sort(gt_ids.begin(), gt_ids.end());

        // 双指针计算交集 
        int match_count = 0;
        // topk_ids 现在就是排序好的视图 topk_row_view
        const auto& topk_ids = topk_row_view; 
        size_t gi = 0, ti = 0;
        while (gi < gt_ids.size() && ti < topk_ids.size()) {
            if (gt_ids[gi] == topk_ids[ti]) {
                ++match_count;
                ++gi;
                ++ti;
            } else if (gt_ids[gi] < topk_ids[ti]) {
                // std::cout << "Query " << i << " missed truth IDs: " << gt_ids[gi] << "\n";
                ++gi;
            } else {
                // std::cout << "Query " << i << " returned false IDs: " << topk_ids[ti] << "\n";
                ++ti;
            }
        }

        // 调试用，输出剩余未匹配的 ground truth
        while (gi < gt_ids.size()) {
            // std::cout << "Query " << i << " missed truth IDs: " << gt_ids[gi] << "\n";
            ++gi;
        }

        accuracy += gt_ids.empty() ? 0.0 : static_cast<double>(match_count) / gt_ids.size();
    }

    if (is_total == true) {
        std::cout << "\n*****Final global accuracy: " << accuracy / calc_size << " *****\n";
    } else {
        std::cout << "Batch accuracy: " << accuracy / calc_size << std::endl;
    }
}


void Batch::allocate(const uint16_t batch_size, const int query_offset,
                     const FlatMatrix<DATA_TYPE>& all_query_vectors,
                     const int qc_num,
                     const FlatMatrix<int32_t>& all_ground_truth)
{
    // 1. 记录批次的元数据
    this->offset = query_offset;
    this->size = batch_size;
    this->qc_num = qc_num;

    // 2. 为当前批次的 vectors 和 ground_truth 分配内存并高效复制数据
    // === vectors ===
    this->vectors.num_vectors = batch_size;
    this->vectors.dimension = all_query_vectors.dim();
    const size_t vectors_elements_to_copy = batch_size * this->vectors.dimension;
    this->vectors.data.resize(vectors_elements_to_copy);

    // 高性能 memcpy: 从全局数据中拷贝一个切片
    const DATA_TYPE* vectors_ptr = all_query_vectors.data.data() + query_offset * all_query_vectors.dim();
    std::memcpy(this->vectors.data.data(), vectors_ptr, vectors_elements_to_copy * sizeof(DATA_TYPE));

    // === ground_truth ===
    this->ground_truth.num_vectors = batch_size;
    this->ground_truth.dimension = all_ground_truth.dim();
    const size_t gt_elements_to_copy = batch_size * this->ground_truth.dimension;
    this->ground_truth.data.resize(gt_elements_to_copy);
    
    // 高性能 memcpy: 从全局数据中拷贝一个切片
    const int32_t* gt_ptr = all_ground_truth.data.data() + query_offset * all_ground_truth.dim();
    std::memcpy(this->ground_truth.data.data(), gt_ptr, gt_elements_to_copy * sizeof(int32_t));

    // 3. 为其他中间结果和输出变量分配扁平化内存
    // === results ===
    this->results.num_vectors = batch_size;
    this->results.dimension = NR_DPUS * MAX_RETURNED_PER_DPU;
    this->results.data.resize(this->results.num_vectors * this->results.dimension);

    // === labels ===
    this->labels.num_vectors = batch_size;
    this->labels.dimension = MAX_QUERY_CLUSTER;
    this->labels.data.resize(this->labels.num_vectors * this->labels.dimension);

    // === var_score_list ===
    this->var_score_list.num_vectors = batch_size;
    this->var_score_list.dimension = NUM_CLUSTER * NUM_CHECKPOINTS;
    this->var_score_list.data.resize(this->var_score_list.num_vectors * this->var_score_list.dimension, 0);

    // === 一维 std::vector ===
    this->split_indices.resize(batch_size, 0);

    // === push / pull buffers ===
    this->broadcast_buffer.resize(QUERY_PUSH_SIZE, 0);
    this->pull_buffers.resize(NR_DPUS, std::vector<uint8_t>(PULL_RESULT_SIZE, 0));

    // 开始对 batch 计时，暂时不在这里写，写在 query_batch.cc 具体执行开始的地方
    // this->start_timer();
}


/** 以下代码暂时不需要 */
// void pre_hinted_query(const std::vector<std::vector<DATA_TYPE>>& query_vectors,
//                       const std::vector<std::vector<int>>& query_cluster_ids, 
//                       const std::vector<std::vector<DATA_TYPE>>& base_vectors,
//                       const std::vector<std::vector<float>>& cluster_var_list, 
//                       const std::vector<uint32_t>& top100_max_dist, 
//                       std::vector<uint32_t>& expected_dist_list, 
//                       const int query_offset, const uint16_t batch_size)
// {
//     for (uint16_t j = 0; j < batch_size; ++j) {
//         uint16_t cid = query_cluster_ids[j][0]; // 取第0个cid，即最近的cluster
//         // 取base_vectors的128个进行计算，可以选前128个，也可以随机选
//         uint32_t random_avg_dist = 0;
//         for (uint16_t i = 0; i < NR_DPUS; ++i) {
//             // int select_id = i;  // 顺序选取最前面的128个
//             int select_id = hash_function(i + random_avg_dist) % cluster_ids[cid].size();  // 随机选取
//             int vec_id = cluster_ids[cid][select_id];
//             random_avg_dist += l2_distance<DATA_TYPE, DATA_TYPE, uint32_t>(query_vectors[j], base_vectors[vec_id]); 
//         }
//         random_avg_dist = random_avg_dist / NR_DPUS;
//         float dist_ratio = static_cast<float>(random_avg_dist) / top100_max_dist[j + query_offset];
//         uint32_t expected_dist = (uint32_t)(static_cast<float>(random_avg_dist) / 1.38);  // 使用hash，最小的一个是1.39292
//         expected_dist_list[j] = std::min(expected_dist, (uint32_t)160000);

//         // std::cout << "Query:" << (j + query_offset) << ", random_avg_dist:" << random_avg_dist 
//         //           << ", dist_ratio:" << dist_ratio 
//         //           << ", expected_dist[" << j << "]:" << expected_dist_list[j] << std::endl;
//     }
// }
