#include "main.hh"

#include <filesystem>


int main(int argc, char* argv[]) 
{
    Config config = parse_args(argc, argv);
    Dataset dataset = load_dataset(config);

    switch (config.query_method) {
        case QueryMethod::MyQuery:
            my_query_fun(config, dataset);
            break;
        case QueryMethod::CpuQuery:
            cpu_query_fun(config, dataset);
            break;
    }

    return 0;
}


// 使用 DPU 加速的查询逻辑
void my_query_fun(const Config& config, const Dataset& dataset) 
{
    auto load_start = std::chrono::steady_clock::now();
    KMeansResult kmeans_result = 
        load_clustering_result(dataset.base_vectors, config.cluster_path, 25, 0.10); 
    auto load_end = std::chrono::steady_clock::now();
    std::cout << "load_clustering_result time: " 
          << std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start).count() 
          << " milliseconds" << std::endl;

    // 加载每个 cluster 的类内 segment 方差，后续先查询方差大的 segment
    auto var_start = std::chrono::steady_clock::now();
    FlatMatrix<float> cluster_var_list = load_intra_cluster_var(
        config.cluster_path, kmeans_result.clusters, dataset.base_vectors, kmeans_result.centroids
    );
    auto var_end = std::chrono::steady_clock::now();
    std::cout << "compute_intra_cluster_var time: " 
          << std::chrono::duration_cast<std::chrono::milliseconds>(var_end - var_start).count() 
          << " milliseconds" << std::endl;

    dpu::DpuSet system = dpu::DpuSet::allocate(NR_DPUS, "nrThreadsPerRank=2");
    system.load("./build/dpu_task");  // dpu 可执行文件相对于运行时的路径
    std::cout << "\n[Launch] Successfully load dpu program." << std::endl;

    auto send_start = std::chrono::steady_clock::now();    
    send_raw_data(system, dataset.base_vectors, kmeans_result.clusters, kmeans_result.multi_labels);  // 发送原始数据到 DPU
    auto send_end = std::chrono::steady_clock::now();
    std::cout << "send_raw_data time: " 
          << std::chrono::duration_cast<std::chrono::milliseconds>(send_end - send_start).count() 
          << " milliseconds" << std::endl;

    const std::filesystem::path path_base(config.nb_running_dpu_log_path);
    const auto& qc_num_sweep = get_qc_num_sweep();
    for (int qc_num : qc_num_sweep) {
        if (qc_num < 1 || qc_num > MAX_QUERY_CLUSTER) {
            throw std::invalid_argument("Sweep qc_num exceeds MAX_QUERY_CLUSTER capacity.");
        }
        const std::string dpu_log_path =
            (path_base.parent_path() /
             (path_base.stem().string() + "_qc" + std::to_string(qc_num) + path_base.extension().string()))
                .string();
        std::cout << "\n[Execute] Performing query with qc_num = " << qc_num << " ..." << std::endl;

        execute_query(system, dataset.query_vectors, cluster_var_list, kmeans_result.centroids,
                      dataset.ground_truth_vids, dpu_log_path, QUERY_MODE, qc_num);
    }

}


// CPU baseline 查询逻辑（使用Faiss-cpu，不采用PQ压缩）
void cpu_query_fun(const Config& config, const Dataset& dataset) 
{
    if ((MIN_REPLICA != 1) or (MAX_REPLICA != 1)) {
        throw std::invalid_argument("ERROR REPLICA for cpu baseline!");
    }
    // 1. 从自定义文件构建 Faiss 索引
    auto load_start = std::chrono::steady_clock::now();
    auto faiss_index = load_index_from_custom_kmeans(config.cluster_path, dataset.base_vectors);
    auto load_end = std::chrono::steady_clock::now();
    std::cout << "load_clustering_result time: " 
          << std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start).count() 
          << " milliseconds" << std::endl;

    // 2. 复用同一个 Faiss 索引，遍历 CPU baseline 的 nprobe
    const auto& qc_num_sweep = get_qc_num_sweep(true);
    for (int qc_num : qc_num_sweep) {
        if (qc_num < 1 || qc_num > MAX_QUERY_CLUSTER) {
            throw std::invalid_argument("Sweep qc_num exceeds MAX_QUERY_CLUSTER capacity.");
        }
        std::cout << "\n[CPU Baseline] Performing query with qc_num = " << qc_num << " ..." << std::endl;
        run_cpu_faiss_query(faiss_index.get(), dataset.query_vectors, dataset.ground_truth_vids, qc_num);
    }
}

