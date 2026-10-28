#include "cluster_var.hh"
#include <omp.h>


FlatMatrix<float> 
load_intra_cluster_var(const std::string& cluster_path, 
                       const JaggedArray<int>& cluster_ids, 
                       const FlatMatrix<DATA_TYPE>& base_vectors,
                       const FlatMatrix<float>& centroids)
{
    std::string var_file; 
    bool file_exist = get_cluster_var_filename(cluster_path, var_file, NUM_CLUSTER);
    
    if (file_exist) {
        std::cout << "\n[Load] Reading intra-cluster variance from: " << var_file << std::endl;
        return load_intra_cluster_var_list(var_file);
    }

    std::cout << "\nIntra-cluster variance file " << var_file 
              << " not found. Recalculating..." << std::endl;
    
    FlatMatrix<float> cluster_var;  // 创建一个空的矩阵用于存储结果
    // 调用计算类内方差函数重新计算，并保存结果，供下次使用
    compute_intra_cluster_var(cluster_ids, base_vectors, centroids, cluster_var);
    save_intra_cluster_var_list(var_file, cluster_var);
    
    std::cout << "Compute and save in-cluster variance done." << std::endl;
    return cluster_var;
}


bool get_cluster_var_filename(const std::string& cluster_path, std::string& filename, int num_cluster) 
{
    if (!std::filesystem::exists(cluster_path)) { 
        // 文件夹不存在则创建目录并返回 false
        std::filesystem::create_directory(cluster_path);
        return false;
    }

    std::string base_name = build_cluster_filename_base(cluster_path, num_cluster);
    filename = base_name + "cluster-var.bin";  // 构建文件名
    // std::cout << "filename:" << filename << std::endl;

    return std::filesystem::exists(filename);
}


void compute_intra_cluster_var(const JaggedArray<int>& cluster_ids, 
                               const FlatMatrix<DATA_TYPE>& base_vectors,
                               const FlatMatrix<float>& centroids, 
                               FlatMatrix<float>& cluster_var)
{
    const int num_clusters = cluster_ids.size();
    const int vector_dim = base_vectors.dim();

    // 初始化输出矩阵
    cluster_var.assign(num_clusters, vector_dim, 0.0f);

    // 使用 OpenMP 并行计算每个簇的方差
    #pragma omp parallel for
    for (int cid = 0; cid < num_clusters; ++cid) {
        auto vector_ids = cluster_ids[cid];
        auto centroid = centroids[cid];
        
        // 获取指向输出矩阵中当前簇对应行的指针
        float* var_vec = cluster_var[cid].data();

        if (vector_ids.empty()) {
            continue; // 跳过空簇
        }

        // 累加平方差
        for (int idx : vector_ids) {
            auto vec = base_vectors[idx];
            for (int d = 0; d < vector_dim; ++d) {
                float diff = static_cast<float>(vec[d]) - centroid[d];
                var_vec[d] += diff * diff;
            }
        }

        // 求均值（方差）
        const float num_members = static_cast<float>(vector_ids.size());
        for (int d = 0; d < vector_dim; ++d) {
            var_vec[d] /= num_members;
        }
    }

    // ===== DEBUG: 打印前5个cluster的方差 =====
    // for (int cid = 0; cid < 5 && cid < num_clusters; ++cid) {
    //     std::cout << "Cluster " << cid << " variances: ";
    //     auto var_view = cluster_var[cid];
    //     for (int d = 0; d < vector_dim; ++d) {
    //         std::cout << var_view[d] << " ";
    //     }
    //     std::cout << std::endl;
    // }
}


void save_intra_cluster_var_list(const std::string& filename, const FlatMatrix<float>& cluster_var) 
{
    std::ofstream out(filename, std::ios::binary);
    if (!out) throw std::runtime_error("Failed to open file for writing: " + filename);

    int32_t num_clusters = cluster_var.size();
    int32_t dim = cluster_var.dim();

    // Write head infomation
    out.write(reinterpret_cast<const char*>(&num_clusters), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(&dim), sizeof(int32_t));

    // Write the entire data block with a single I/O call
    out.write(reinterpret_cast<const char*>(cluster_var.data.data()), 
              cluster_var.data.size() * sizeof(float));
    out.close();
}


FlatMatrix<float> load_intra_cluster_var_list(const std::string& filename) 
{
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open file for reading: " + filename);
    }

    int32_t num_clusters = 0;
    int32_t dim = 0;
    
    // Read in head infomation
    if (!in.read(reinterpret_cast<char*>(&num_clusters), sizeof(int32_t)) ||
        !in.read(reinterpret_cast<char*>(&dim), sizeof(int32_t))) {
        throw std::runtime_error("Failed to read header from file: " + filename);
    }

    assert(num_clusters == NUM_CLUSTER);
    assert(dim == VECTOR_DIM);

    // Prepare the FlatMatrix object
    FlatMatrix<float> var_matrix(num_clusters, dim);
    const size_t total_elements = static_cast<size_t>(num_clusters) * dim;

    // Read the entire data block with a single I/O call
    const size_t total_bytes = total_elements * sizeof(float);
    if (!in.read(reinterpret_cast<char*>(var_matrix.data.data()), total_bytes)) {
        throw std::runtime_error("File truncated or read error in: " + filename);
    }
    
    in.close();
    return var_matrix;
}