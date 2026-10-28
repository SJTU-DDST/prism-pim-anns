#include "query_batch_cpu.hh"
#include "../data_utils/kmeans.hh"
#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <omp.h>


std::shared_ptr<faiss::IndexIVFFlat> 
load_index_from_custom_kmeans(const std::string& cluster_path,
                              const FlatMatrix<DATA_TYPE>& database_vectors)
{
    std::string cluster_file; 
    bool file_exist = get_cluster_filename(cluster_file, cluster_path, 
                                           NUM_CLUSTER, 0.10, MIN_REPLICA, MAX_REPLICA);

    if (file_exist) {
        std::cout << "\n[Load] Reading K-means results from: " << cluster_file << std::endl;
    } else {
        std::cout << "\nCannot find K-means results from: " << cluster_file << std::endl;
        exit(1);
    }

    std::ifstream infile(cluster_file, std::ios::binary);
    if (!infile.is_open()) {
        throw std::runtime_error("Cannot open file: " + cluster_file + " for reading.");
    }

    // 1. 读取文件头
    uint64_t k, d, n;
    infile.read(reinterpret_cast<char*>(&k), sizeof(k));
    infile.read(reinterpret_cast<char*>(&d), sizeof(d));
    infile.read(reinterpret_cast<char*>(&n), sizeof(n));
    if (!infile) {
        throw std::runtime_error("Failed to read header from binary file: " + cluster_file);
    }

    std::cout << "Loading K-means file: K=" << k << ", D=" << d << ", N=" << n << std::endl;

    // 2. 读取质心数据
    std::vector<float> centroids_data(k * d);
    infile.read(reinterpret_cast<char*>(centroids_data.data()), k * d * sizeof(float));
    if (!infile) {
        throw std::runtime_error("Failed to read centroids from binary file: " + cluster_file);
    }

    // --- 3. 读取倒排列表结构 (offsets 和 member IDs) ---
    // 首先读取 offsets 数组
    std::vector<uint64_t> offsets(k + 1);
    infile.read(reinterpret_cast<char*>(offsets.data()), (k + 1) * sizeof(uint64_t));
    if (!infile) {
        throw std::runtime_error("Failed to read offsets from binary file: " + cluster_file);
    }

    // *** 关键修正点：使用 offsets[k] 来确定成员ID的总数 ***
    uint64_t total_members = offsets[k];
    std::cout << "Total members in inverted lists: " << total_members << std::endl;
    std::vector<int32_t> member_ids(total_members);
    infile.read(reinterpret_cast<char*>(member_ids.data()), total_members * sizeof(int32_t));
    if (!infile) {
        throw std::runtime_error("Failed to read member_ids from binary file: " + cluster_file);
    }
    infile.close();
    
    // --- 使用读取的数据构建 Faiss 索引 ---
    // a. 创建量化器 (quantizer) 并添加质心
    auto quantizer = new faiss::IndexFlatL2(d); // 使用 L2 距离
    quantizer->add(k, centroids_data.data());

    // b. 创建 IndexIVFFlat 实例
    // 注意：我们将量化器交给 IndexIVFFlat 管理，所以将 own_fields 设置为 true
    auto index = std::make_shared<faiss::IndexIVFFlat>(quantizer, d, k, faiss::METRIC_L2);
    index->own_fields = true;
    index->is_trained = true; // 我们是手动加载的，所以标记为已训练

    // c. 手动填充倒排列表 (Inverted Lists)
    // 这是最关键的步骤。我们需要遍历每个 cluster，
    // 找到属于它的所有向量的 ID，然后从原始数据库中获取这些向量的数据，
    // 最后将 (ID, vector_data) 对添加到 Faiss 的倒排列表中。
    std::cout << "Populating inverted lists..." << std::endl;
    for (size_t i = 0; i < k; ++i) {
        size_t list_size = offsets[i+1] - offsets[i];
        if (list_size == 0) continue;

        // 获取当前 cluster 的所有成员 ID
        const int32_t* id_ptr = &member_ids[offsets[i]];
        
        // 准备要添加到 Faiss 的数据
        std::vector<faiss::idx_t> ids_to_add(list_size);
        std::vector<float> vectors_to_add(list_size * d);

        for (size_t j = 0; j < list_size; ++j) {
            int32_t original_vec_id = id_ptr[j];
            ids_to_add[j] = static_cast<faiss::idx_t>(original_vec_id);

            // 使用 operator[] 获取代表向量的 span
            std::span<const DATA_TYPE> vec_span = database_vectors[static_cast<size_t>(original_vec_id)];

            // 将 span 中的数据逐个转换为 float
            float* vector_dest = vectors_to_add.data() + j * d;
            for (size_t dim = 0; dim < d; ++dim) {
                // 直接通过 span 访问元素
                vector_dest[dim] = static_cast<float>(vec_span[dim]);
            }
        }

        // 使用底层 API `add_entries` 将数据直接添加到指定的倒排列表
        index->invlists->add_entries(
            i, // 列表编号 (list_no)
            list_size,
            ids_to_add.data(),
            reinterpret_cast<const uint8_t*>(vectors_to_add.data())
        );
    }
    index->ntotal = n; // 更新索引中的向量总数

    std::cout << "Successfully built Faiss index from custom K-means file." << std::endl;
    return index;
}


long long 
calculate_batch_accuracy(const FlatMatrix<int32_t>& ground_vids, 
                         const std::vector<faiss::idx_t>& result_ids, 
                         const int offset, 
                         const int batch_size)
{
    // 现在这个变量代表：在当前批次中，所有查询找到的真实近邻ID的总数（|R ∩ A| 的总和）
    long long total_found_count = 0; 
    
    for (int i = 0; i < batch_size; ++i) {
        // ground_vids[offset + i] 是第 i 个查询的真实近邻集合 R
        const auto& ground_truth_set = ground_vids[offset + i];
        
        // 遍历 TOP_K 结果集合 A 中的每个返回ID
        for (int k = 0; k < TOP_K; ++k) {
            // Faiss 返回的结果是 1D 数组，需要计算索引
            faiss::idx_t result_id = result_ids[i * TOP_K + k];
            
            // 检查返回的ID是否存在于真实近邻集合 R 中
            if (std::find(ground_truth_set.begin(), ground_truth_set.end(), result_id) != ground_truth_set.end()) {
                // 找到了一个真实近邻，计数器加 1
                total_found_count++;
                // 我们要计算的是 |R ∩ A|，即要统计所有被正确找回的近邻，而不是找到一个就停止。
            }
        }
    }
    // 返回批次中找到的所有真实近邻的总数
    return total_found_count;
}


void run_cpu_faiss_query(faiss::Index* index, 
                         const FlatMatrix<DATA_TYPE>& query_vectors, 
                         const FlatMatrix<int32_t>& ground_vids,
                         int qc_num)
{
    int query_offset = 0;
    const int total_query_num = query_vectors.size();
    const int total_batches = (total_query_num + BATCH_SIZE - 1) / BATCH_SIZE;
    long long total_correct_count = 0;

    HostTimingStats host_stats;

    // 在 run_cpu_faiss_query 函数的开头
    faiss::IndexIVF* ivf_index = dynamic_cast<faiss::IndexIVF*>(index);

    if (!ivf_index) {
        // 如果转换失败，意味着传入的不是一个 IVF 类型的索引，它没有 nprobe。
        // 我们可以抛出异常或打印错误信息后返回。
        std::cerr << "Error: The provided Faiss index is not an IVF index, so 'nprobe' is not accessible." << std::endl;
        return;
    }

    // 可以使用 ivf_index 来安全地设置 nprobe
    ivf_index->nprobe = qc_num;

    auto global_start_time = std::chrono::steady_clock::now();

    while (query_offset < total_query_num) {
        int remaining_num = total_query_num - query_offset;
        const uint16_t batch_size = std::min(BATCH_SIZE, remaining_num);
        // std::cout << "[Query] query_offset: " << query_offset << ", batch_size " << batch_size 
        //           << ", remaining_num: " << remaining_num << std::endl;

        auto batch = std::make_shared<Batch>();
        // 假设 allocate 仅从 query_vectors 中提取数据到 batch->vectors
        batch->allocate(batch_size, query_offset, query_vectors, qc_num, ground_vids);

        std::vector<float> query_vectors_float(batch_size * VECTOR_DIM);  // 将查询向量转化为 Faiss 所需的 float 类型
        for (uint16_t i = 0; i < batch_size; ++i) {
            // 获取当前查询向量在总查询集中的索引
            size_t query_idx = query_offset + i;
            
            // 获取第 query_idx 个向量的 span 视图
            std::span<const DATA_TYPE> query_vec_span = query_vectors[query_idx];

            // 遍历该向量的每个维度进行转换
            for (size_t j = 0; j < VECTOR_DIM; ++j) {
                query_vectors_float[i * VECTOR_DIM + j] = static_cast<float>(query_vec_span[j]);
            }
        }

        // 为 Faiss 的返回结果分配空间
        std::vector<faiss::idx_t> result_ids(batch_size * TOP_K);
        std::vector<float> result_dists(batch_size * TOP_K);
        
        // 1. & 2. & 3. 合并为一步：使用 Faiss 进行搜索
        // Faiss 的 index->search 会自动处理 finding clusters 和 searching within them
        {
            ScopedTimer timer(host_stats.query_time);
            index->search(batch_size, query_vectors_float.data(), TOP_K, result_dists.data(), result_ids.data());
        }

        // 4. 计算批次准确率，结果已经在 result_ids 中
        {
            ScopedTimer timer(host_stats.pull_time); // 可以重用这个计时器来测量准确率计算时间

            // 1. 调用修改后的函数，计算找到的真实近邻总数
            long long batch_found_neighbors = calculate_batch_accuracy(ground_vids, result_ids, query_offset, batch_size);

            // 2. 累加找到的总数（用于最后计算总体的平均 Recall@K）
            total_correct_count += batch_found_neighbors; 

            // 3. 计算分母：当前批次中所有查询的真实近邻总数（即 |R| 的总和）
            long long total_ground_truth_in_batch = static_cast<long long>(batch_size) * TOP_K;

            // 4. 计算标准的 Batch Recall@K：(|R ∩ A| 的总和) / (|R| 的总和)
            double batch_recall = 0.0;
            if (total_ground_truth_in_batch > 0) {
                batch_recall = static_cast<double>(batch_found_neighbors) / total_ground_truth_in_batch;
            }

            // 输出批次 Recall
            // std::cout << "[Batch " << (query_offset / BATCH_SIZE) << "] Recall@" << TOP_K << ": " << batch_recall << std::endl;
        }
        query_offset += batch_size;
    }

    host_stats.global_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - global_start_time
    );

    host_stats.global_time = host_stats.query_time;  // 只统计具体的 query time

    // 计算并打印最终的总准确率以及统计的时间
    host_stats.print(total_query_num, total_batches);

    double total_recall = static_cast<double>(total_correct_count) / total_query_num / TOP_K;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "[CPU Baseline] Final Recall@" << TOP_K << ": " << total_recall 
              << " (" << total_correct_count << "/" << (total_query_num * TOP_K) << ")" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
}


// 这个效果开32线程跟上面的一样，事实上上面简单的写法就已经调用了多线程
void run_cpu_faiss_query_multithreaded(faiss::Index* index,
                                       const FlatMatrix<DATA_TYPE>& query_vectors,
                                       const FlatMatrix<int32_t>& ground_vids)
{
    const int total_query_num = query_vectors.size();
    HostTimingStats host_stats;

    // --- 1. 一次性准备所有查询数据 ---
    // 去掉 while 循环，一次性将所有查询向量从 DATA_TYPE 转换为 float
    std::cout << "[Data Prep] Preparing " << total_query_num << " query vectors for Faiss..." << std::endl;
    std::vector<float> all_query_vectors_float(static_cast<size_t>(total_query_num) * VECTOR_DIM);

    int num_threads = 32;  // 设置Faiss内部使用的线程数
    omp_set_num_threads(num_threads);
    std::cout << "[Info] Faiss is configured to use " << omp_get_max_threads() << " threads." << std::endl;

    // 这个数据准备阶段本身也可以并行化，以进一步缩短预处理时间
    #pragma omp parallel for
    for (int i = 0; i < total_query_num; ++i) {
        std::span<const DATA_TYPE> query_vec_span = query_vectors[i];
        for (size_t j = 0; j < VECTOR_DIM; ++j) {
            all_query_vectors_float[static_cast<size_t>(i) * VECTOR_DIM + j] = static_cast<float>(query_vec_span[j]);
        }
    }
    std::cout << "[Data Prep] Data preparation complete." << std::endl;

    // --- 3. 设置索引参数 (nprobe) ---
    faiss::IndexIVF* ivf_index = dynamic_cast<faiss::IndexIVF*>(index);
    if (ivf_index) {
        ivf_index->nprobe = MAX_QUERY_CLUSTER;
        std::cout << "[Info] Index nprobe set to " << ivf_index->nprobe << std::endl;
    } else {
        std::cout << "[Warning] The index is not an IVF index. 'nprobe' parameter is not applicable." << std::endl;
    }

    // --- 4. 执行一次大规模的并行搜索 ---
    std::vector<faiss::idx_t> result_ids(static_cast<size_t>(total_query_num) * TOP_K);
    std::vector<float> result_dists(static_cast<size_t>(total_query_num) * TOP_K);

    std::cout << "[Search] Starting Faiss search for " << total_query_num << " queries..." << std::endl;
    auto global_start_time = std::chrono::steady_clock::now();
    
    // 核心调用：将所有查询一次性交给 Faiss，Faiss 内部会自动使用 OpenMP 并行处理
    index->search(total_query_num, all_query_vectors_float.data(), TOP_K, result_dists.data(), result_ids.data());

    host_stats.global_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - global_start_time
    );

    // --- 5. 计算总的准确率 (Recall@K) ---
    long long total_correct_count = 0;
    // 准确率计算也可以并行
    #pragma omp parallel for reduction(+:total_correct_count)
    for (int i = 0; i < total_query_num; ++i) {
        std::span<const int32_t> ground_truth_span = ground_vids[i];
        std::unordered_set<int32_t> ground_truth_set(ground_truth_span.begin(), ground_truth_span.end());
        
        long long correct_in_query = 0;
        for (int k = 0; k < TOP_K; ++k) {
            if (ground_truth_set.count(result_ids[static_cast<size_t>(i) * TOP_K + k])) {
                correct_in_query++;
            }
        }
        total_correct_count += correct_in_query;
    }

    // --- 6. 打印最终结果 ---
    long long total_ground_truth_count = static_cast<long long>(total_query_num) * ground_vids[0].size(); // 所有gt大小相同
    double total_recall = total_ground_truth_count > 0 ? static_cast<double>(total_correct_count) / total_ground_truth_count : 0.0;


    // 计算并打印最终的总准确率以及统计的时间
    host_stats.print(total_query_num, (int)(total_query_num / BATCH_SIZE + 1));
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "[CPU Multi-threaded] Final Recall@" << TOP_K << ": " << total_recall
              << " (" << total_correct_count << "/" << total_ground_truth_count << ")" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
}
