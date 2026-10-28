#include "kmeans.hh"
#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <omp.h>  // 聚类过程中使用多线程
#include <charconv>  // C++17 for std::from_chars


KMeansResult 
load_clustering_result(const FlatMatrix<DATA_TYPE>& vectors, 
                       const std::string& cluster_path, 
                       int kmeans_iterations = 25, float epsilon = 0.10) 
{
    std::string cluster_file; 
    bool file_exist = get_cluster_filename(cluster_file, cluster_path, 
                                           NUM_CLUSTER, epsilon, MIN_REPLICA, MAX_REPLICA);

    if (file_exist) {
        std::cout << "\n[Load] Reading K-means results from: " << cluster_file << std::endl;
        KMeansResult result = load_kmeans_results(cluster_file);
        return result;
    }

    std::cout << "\nK-means results file " << cluster_file 
              << " not found. Recalculating..." << std::endl;
    
    KMeansResult result;  // 创建一个KMeansResult对象来存储结果
    // 调用聚类函数重新聚类，它会填充 result.multi_labels 和 result.centroids
    faiss_kmeans_with_duplication(vectors, result.multi_labels, result.centroids, kmeans_iterations, epsilon);
    result.K = result.centroids.size();

    // 根据刚刚计算出的 multi_labels，反向构建 clusters
    std::cout << "Inverting multi_labels to create clusters efficiently..." << std::endl;
    create_clusters_from_multi_labels(result.clusters, result.multi_labels, result.K);

    save_kmeans_results(cluster_file, result);  // 保存聚类结果

    std::cout << "Compute and save K-means results done." << std::endl;
    return result;
}


bool get_cluster_filename(std::string& filename, const std::string& cluster_path, 
                          int num_cluster, float epsilon, int min_dup, int max_dup) 
{
    if (!std::filesystem::exists(cluster_path)) { 
        // 文件夹不存在则创建目录并返回 false
        std::filesystem::create_directory(cluster_path);
        return false;
    }

    // 调用基础函数获取通用前缀
    std::string base_name = build_cluster_filename_base(cluster_path, num_cluster);
    
    std::ostringstream suffix_ss;  // 拼接后缀
    suffix_ss << "cluster-" << std::fixed << std::setprecision(3) << epsilon
              << "-min" << min_dup << "-max" << max_dup << ".bin";
              
    filename = base_name + suffix_ss.str();
    // std::cout << "filename:" << filename << std::endl;

    return std::filesystem::exists(filename);
}


void faiss_kmeans_with_duplication(const FlatMatrix<DATA_TYPE>& data,
                                   JaggedArray<int>& multi_labels, 
                                   FlatMatrix<float>& centroids, 
                                   int iterations = 25, float epsilon = 0.1) 
{
    const int N = data.size();
    const int D = data.dim();
    
    if (N < NUM_CLUSTER) {
        throw std::runtime_error("Error: Number of vectors (" + std::to_string(N) + 
                                 ") is less than clusters (" + std::to_string(NUM_CLUSTER) + ").");
    }

    // faiss聚类，启用多线程
    int num_threads = 16;
    std::cout << "Performing K-means for dataset. Current clustering (OpenMP) threads: " 
              << num_threads << std::endl;
    omp_set_num_threads(num_threads);

    // 为FAISS准备数据，数据采用展平的形式
    auto deleter = [](float* p) { free(p); };
    std::unique_ptr<float[], decltype(deleter)> flat_data_ptr(
        (float*)malloc(sizeof(float) * N * D), 
        deleter
    );

    if (!flat_data_ptr) {
        throw std::runtime_error("Failed to allocate memory for FAISS data.");
    }
    float* flat_data = flat_data_ptr.get(); 

    // 并行地将输入数据转换为 float 类型
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        const auto data_row = data[i];
        float* flat_row = flat_data + (size_t)i * D;
        for (int j = 0; j < D; j++) {
            flat_row[j] = static_cast<float>(data_row[j]);
        }
    }

    // FAISS 聚类
    faiss::ClusteringParameters cp;
    cp.niter = iterations;
    faiss::IndexFlatL2 centroid_index(D); 
    faiss::Clustering clustering(D, NUM_CLUSTER, cp);
    clustering.train(N, flat_data, centroid_index);

    const int top_k = MAX_REPLICA;  // 这里的 topk 是 vector 距离最近的 cluster 质心的数量
    
    // 创建一个大小为 N 的临时容器来并行收集结果
    std::vector<std::vector<int>> all_labels(N);

    const int batch_size = (N > 1'000'000) ? 500000 : N;  // 按批次处理数据
    for (int offset = 0; offset < N; offset += batch_size) {
        int cur_batch_size = std::min(batch_size, N - offset);
        // Faiss search，查询每个向量距离最近的若干质心
        std::vector<faiss::idx_t> nearest_centroids(cur_batch_size * top_k);
        std::vector<float> centroid_distances(cur_batch_size * top_k);
        centroid_index.search(cur_batch_size, flat_data + (size_t)offset * D, top_k,
                                centroid_distances.data(), nearest_centroids.data());

        #pragma omp parallel for
        for (int i = 0; i < cur_batch_size; i++) {
            const float d = centroid_distances[(size_t)i * top_k];  // 最近的 topk 个质心
            std::vector<int> local_labels;  // 使用局部的 local_labels
            local_labels.reserve(top_k);  // 预分配空间，避免pushback扩容

            for (int j = 0; j < top_k; j++) {
                // 如果已满足 MAX_REPLICA，终止循环
                if (local_labels.size() >= MAX_REPLICA) break;

                float d_j = centroid_distances[(size_t)i * top_k + j];  // 距离第 j 个最近质心的距离

                bool should_add_label;
                #if (MAX_REPLICA == MIN_REPLICA)
                    // MIN_REPLICA 和 MAX_REPLICA 相等，此时不存在对边界节点的单独处理
                    should_add_label = (local_labels.size() < MIN_REPLICA);
                #else
                    // 对于边界节点，应该根据距离判断结果，复制至多 MAX_REPLICA 份
                    should_add_label = (local_labels.size() < MIN_REPLICA || d_j <= (1 + epsilon) * d);
                #endif

                if (should_add_label) {
                    // 将 vector 添加到该质心所属的类中
                    local_labels.push_back(static_cast<int>(nearest_centroids[(size_t)i * top_k + j]));
                } else {
                    break;
                }
            }
            all_labels[offset + i] = std::move(local_labels);  // 直接写入总容器的对应位置，这是线程安全的
        }
    }

    // 在所有并行计算结束后，用一个简单串行的循环来构建最终的 JaggedArray。
    std::cout << "Finalizing multi_labels from " << all_labels.size() << " entries..." << std::endl;
    multi_labels.data.clear();
    multi_labels.offsets.clear();
    multi_labels.offsets.reserve(N + 1);

    // 可以先精确计算总大小并预分配内存，以获得最佳性能
    size_t total_data_size = 0;
    for (const auto& labels : all_labels) {
        total_data_size += labels.size();
    }
    multi_labels.data.reserve(total_data_size);

    // 开始构建
    multi_labels.offsets.push_back(0);
    for (const auto& labels : all_labels) {
        multi_labels.data.insert(multi_labels.data.end(), labels.begin(), labels.end());
        multi_labels.offsets.push_back(multi_labels.data.size());
    }

    if (multi_labels.size() != N) {
        throw std::runtime_error("Error: Final multi_labels size (" + std::to_string(multi_labels.size()) 
                                 + ") does not match N (" + std::to_string(N) + ").");
    }

    std::cout << "Will start recalculating centroides ..." << std::endl;

    // 质心重计算
    centroids.num_vectors = NUM_CLUSTER;
    centroids.dimension = D;
    centroids.data.assign(NUM_CLUSTER * D, 0.0f);
    std::vector<int> cluster_counts(NUM_CLUSTER, 0);

    // 第一段并行：累加数据和计数
    #pragma omp parallel
    {
        // 每个线程创建本地临时变量避免竞争
        std::vector<float> local_centroids_data(NUM_CLUSTER * D, 0.0f);
        std::vector<int> local_counts(NUM_CLUSTER, 0);

        // 并行处理每个数据点
        #pragma omp for nowait  // nowait 表示不需同步等待
        for (int i = 0; i < N; ++i) {
            auto data_row = data[i];
            auto label_row = multi_labels[i];
            for (int cid : label_row) {
                local_counts[cid]++;
                for (int d = 0; d < D; ++d) {
                    local_centroids_data[(size_t)cid * D + d] += data_row[d];
                }
            }
        }

        // 合并各线程结果到主变量（需要临界区保护）
        #pragma omp critical
        {
            for (int cid = 0; cid < NUM_CLUSTER; ++cid) {
                cluster_counts[cid] += local_counts[cid];
                for (int d = 0; d < D; ++d) {
                    centroids.data[(size_t)cid * D + d] += local_centroids_data[(size_t)cid * D + d];
                }
            }
        }
    }

    // 第二段并行：计算平均值（无数据竞争可直接并行）
    #pragma omp parallel for
    for (int i = 0; i < NUM_CLUSTER; ++i) {
        if (cluster_counts[i] > 0) {
            float* centroid_row = centroids[i].data();
            for (int d = 0; d < D; ++d) {
                centroid_row[d] /= cluster_counts[i];
            }
        }
    }
    
    omp_set_num_threads(1);  // 恢复为单线程
    std::cout << "Number of labels (with duplication): " << multi_labels.data.size() << std::endl;
}


void save_kmeans_results(const std::string& filename, const KMeansResult& result)
{
    // 以二进制模式打开文件
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename + " for writing.");
    }

    // 1. 准备并写入文件头
    const uint64_t k = result.K;
    const uint64_t d = result.centroids.dimension;
    const uint64_t n = result.multi_labels.size();  // 数据点总数可以从 multi_labels 的行数中获取
    
    outfile.write(reinterpret_cast<const char*>(&k), sizeof(k));
    outfile.write(reinterpret_cast<const char*>(&d), sizeof(d));
    outfile.write(reinterpret_cast<const char*>(&n), sizeof(n));

    // 2. 直接写入质心数据
    // FlatMatrix 的内部数据是连续的，非常适合直接写入。
    outfile.write(reinterpret_cast<const char*>(result.centroids.data.data()),
                  result.centroids.data.size() * sizeof(float));

    // 3. 直接写入 `clusters` 的内部数据    
    // 写入偏移量数组
    outfile.write(reinterpret_cast<const char*>(result.clusters.offsets.data()),
                  result.clusters.offsets.size() * sizeof(uint64_t));

    // 写入成员ID数据
    outfile.write(reinterpret_cast<const char*>(result.clusters.data.data()),
                  result.clusters.data.size() * sizeof(int32_t)); // 确保数据类型与读取时一致

    if (!outfile) {
        throw std::runtime_error("An error occurred while writing to the binary file.");
    }
    
    outfile.close();
    std::cout << "K-means results successfully saved to binary file: " << filename << std::endl;
}


KMeansResult load_kmeans_results(const std::string& filename) 
{
    std::ifstream infile(filename, std::ios::binary);
    if (!infile) {
        throw std::runtime_error("Cannot open binary file: " + filename);
    }

    KMeansResult result;

    // 1. 读取文件头
    uint64_t k, d, n;
    infile.read(reinterpret_cast<char*>(&k), sizeof(k));
    infile.read(reinterpret_cast<char*>(&d), sizeof(d));
    infile.read(reinterpret_cast<char*>(&n), sizeof(n));
    if (!infile) {
        throw std::runtime_error("Failed to read header from binary file.");
    }
    result.K = static_cast<int>(k);

    // 2. 直接读取质心数据
    result.centroids.num_vectors = k;
    result.centroids.dimension = d;
    result.centroids.data.resize(k * d);
    infile.read(reinterpret_cast<char*>(result.centroids.data.data()), k * d * sizeof(float));

    // --- 3. 直接加载 `result.clusters` ---
    // 直接读取偏移量到 JaggedArray 的 offsets 向量中
    result.clusters.offsets.resize(k + 1);
    infile.read(reinterpret_cast<char*>(result.clusters.offsets.data()), (k + 1) * sizeof(uint64_t));
    
    // 直接读取成员ID到 JaggedArray 的 data 向量中
    uint64_t total_members = result.clusters.offsets[k];
    result.clusters.data.resize(total_members);
    infile.read(reinterpret_cast<char*>(result.clusters.data.data()), total_members * sizeof(int32_t));
    
    if (!infile) throw std::runtime_error("Failed to read clusters data from binary file.");
    infile.close();

    // 高效构建 `result.multi_labels`
    // 这里需要计算，使用两遍法（计数+放置）来避免动态内存分配
    std::cout << "[Reconstruction] Reconstruction data labels ..." << std::endl;
    
    // 遍一：计数。计算每个数据点属于多少个聚类。
    std::vector<uint32_t> point_counts(n, 0);
    for (const auto& member_id : result.clusters.data) {
        if (static_cast<uint64_t>(member_id) < n) {
            point_counts[member_id]++;
        }
    }

    // 根据计数构建 `multi_labels` 的偏移量数组 (前缀和)
    result.multi_labels.offsets.resize(n + 1);
    result.multi_labels.offsets[0] = 0;
    for (uint64_t i = 0; i < n; ++i) {
        result.multi_labels.offsets[i+1] = result.multi_labels.offsets[i] + point_counts[i];
    }

    // 预分配 `multi_labels` 的数据区
    uint64_t total_multi_labels_entries = result.multi_labels.offsets[n];
    result.multi_labels.data.resize(total_multi_labels_entries);

    // 遍二：放置。再次遍历 clusters，将 cluster_id 放置到 multi_labels.data 的正确位置。
    // point_counts 现在被当作写入位置的动态游标。
    for (int cluster_id = 0; cluster_id < result.K; ++cluster_id) {
        auto members_of_cluster = result.clusters[cluster_id]; // 使用 operator[] 获取数据视图
        for (int member_id : members_of_cluster) {
            if (static_cast<uint64_t>(member_id) < n) {
                uint64_t index_in_data = result.multi_labels.offsets[member_id] + (--point_counts[member_id]);
                result.multi_labels.data[index_in_data] = cluster_id;
            }
        }
    }
    
    std::cout << "[Load done] K-means results fully loaded." << std::endl;
    return result;
}


void create_clusters_from_multi_labels(JaggedArray<int>& clusters, const JaggedArray<int>& multi_labels, int K)
{
    // 第一次遍历：计数。计算每个聚类中有多少个成员。
    std::vector<uint32_t> cluster_counts(K, 0);
    for (const auto& cluster_id : multi_labels.data) {
        if (cluster_id < K) {
            cluster_counts[cluster_id]++;
        }
    }

    // 根据计数构建 `clusters` 的偏移量数组 (前缀和)
    clusters.offsets.resize(K + 1);
    clusters.offsets[0] = 0;
    for (int i = 0; i < K; ++i) {
        clusters.offsets[i+1] = clusters.offsets[i] + cluster_counts[i];
    }

    // 预分配 `clusters` 的数据区
    uint64_t total_members = clusters.offsets[K];
    clusters.data.resize(total_members);

    // 第二次遍历：放置。遍历 multi_labels，将成员ID（vidx）放置到 clusters.data 的正确位置。
    // cluster_counts 现在被当作写入位置的动态游标。
    for (size_t vidx = 0; vidx < multi_labels.size(); ++vidx) {
        auto assigned_clusters = multi_labels[vidx]; // 获取该点所属的聚类列表
        for (int cluster_id : assigned_clusters) {
            if (cluster_id < K) {
                uint64_t index_in_data = clusters.offsets[cluster_id] + (--cluster_counts[cluster_id]);
                clusters.data[index_in_data] = static_cast<int>(vidx);
            }
        }
    }
}

