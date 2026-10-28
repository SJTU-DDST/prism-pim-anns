#ifndef QUERY_PBATCH_CPU_HH
#define QUERY_BATCH_CPU_HH
#include <queue>
#include <memory>  // 用于 std::shared_ptr
#include <cmath>
#include <mutex>

#include "host_base.hh"

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h> // 需要这个头文件来进行 dynamic_cast
#include <faiss/IndexIVFFlat.h>


/**
 * @brief 从自定义的 K-means 二进制结果文件中加载数据来构建一个 Faiss IndexIVFFlat。
 * @param kmeans_filename 您的自定义二进制文件名。
 * @param database_vectors 包含所有原始数据库向量的连续浮点数组。
 * @return std::shared_ptr<faiss::IndexIVFFlat> 一个构建好的、可供搜索的 Faiss 索引。
 */
std::shared_ptr<faiss::IndexIVFFlat> 
load_index_from_custom_kmeans(const std::string& cluster_path,
                              const FlatMatrix<DATA_TYPE>& database_vectors);


/**
 * @brief 计算一个批次的查询准确率 (Recall@K)
 * @param ground_vids 真实近邻的ID
 * @param topk_results Faiss 返回的 top-K 结果
 * @param offset 当前批次在总查询中的偏移
 * @param batch_size 批次大小
 * @return 批次中命中真实近邻的查询数量
 */
long long 
calculate_batch_accuracy(const FlatMatrix<int32_t>& ground_vids, 
                         const std::vector<faiss::idx_t>& result_ids, 
                         const int offset, 
                         const int batch_size);


/**
 * @brief 使用 Faiss 在 CPU 上运行 ANNS 查询的基线版本
 * @param index 一个已经构建和训练好的 Faiss 索引
 * @param query_vectors 查询向量
 * @param ground_truth_vids 真实 topk 向量的 id
 * @param qc_num 当前这轮查询的 nprobe
 */
void run_cpu_faiss_query(faiss::Index* index, 
                         const FlatMatrix<DATA_TYPE>& query_vectors, 
                         const FlatMatrix<int32_t>& ground_vids,
                         int qc_num = MAX_QUERY_CLUSTER);


void run_cpu_faiss_query_multithreaded(faiss::Index* index,
                                       const FlatMatrix<DATA_TYPE>& query_vectors,
                                       const FlatMatrix<int32_t>& ground_vids);



#endif // QUERY_BATCH_CPU_HH
