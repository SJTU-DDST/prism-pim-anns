#ifndef HOST_BASE_HH
#define HOST_BASE_HH
#include <vector>
#include <cstdint>
#include <queue>
#include <unordered_set>
#include <cmath>
#include <memory>

#include "dpu_host_xfer.hh"


// 每一个查询批次的结构体
struct Batch {
    FlatMatrix<DATA_TYPE> vectors;  // 当前批次的查询向量
    FlatMatrix<int32_t> ground_truth;  // 当前批次的真实标签ID
    FlatMatrix<std::pair<uint32_t, int>> results;  // 从DPU返回的原始结果
    FlatMatrix<int> labels;  // 每个查询向量被分配到的聚类ID
    // segment 方差的列表，维度: num_vectors=batch_size, dimension=(NUM_CLUSTER * NUM_CHECKPOINTS)
    FlatMatrix<uint8_t> var_score_list; 

    int qc_num;  // 本批次所有查询共用的聚类数量
    std::vector<uint8_t> split_indices;  // 用于标识全部查询和只查询交集部分的分界 index

    int offset;  // 当前批次在总查询中的起始偏移
    uint16_t size;  // 当前批次的大小

    std::vector<uint8_t> broadcast_buffer;  // host 发送给 DPU query 的缓冲区
    std::vector<std::vector<uint8_t>> pull_buffers;  // 从 DPU 收回 results 的缓冲区

    // 新增计时成员
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point end_time;
    DurationType processing_time{DurationType::zero()};  // 处理时间（微秒）

    // allocate 方法，构造一个 batch
    void allocate(const uint16_t batch_size, const int query_offset,
                  const FlatMatrix<DATA_TYPE>& all_query_vectors,
                  const int qc_num,
                  const FlatMatrix<int32_t>& all_ground_truth);

    // 新增计时方法
    void start_timer() {
        start_time = std::chrono::steady_clock::now();
    }
    
    void stop_timer() {
        end_time = std::chrono::steady_clock::now();
        processing_time = std::chrono::duration_cast<DurationType>(end_time - start_time);
    }
    
    // 返回微秒时间
    DurationType get_processing_time() const {
        return processing_time;
    }
    
    // 转换为指定单位（暂时无用）
    template<typename T>
    auto get_processing_time_as() const {
        return std::chrono::duration_cast<T>(processing_time).count();
    }
};


/**
 * @brief 计算 query vector 在每个 cluster 下不同维度 segment 中的方差，得出 score。
 *        后续查询会从 score 大的 segment 开始计算，有利于更快达到阈值提前退出计算，减少计算量。
 *        方差包括类内（cluster 中数据到 centroid）方差，和类间（query vector 到 centroids）方差。
 * @param query_vectors 查询向量 
 * @param centroids 质心向量 
 * @param query_clusters 每个向量待查询的 clusters 列表 
 * @param cluster_var_list 每个 cluster 中数据的方差列表（类内方差列表） 
 * @param var_score_list 方差得分列表，维度是 [batch][cluster][max_dims_start]
 * @param qc_num 本批次所有查询共用的聚类数量
 * @param alpha 类内方差加权系数
 * @param beta 类间方差加权系数
 */
void compute_query_segment_scores(const FlatMatrix<DATA_TYPE>& query_vectors, 
                                 const FlatMatrix<float>& centroids, 
                                 const FlatMatrix<int>& query_clusters, 
                                 const FlatMatrix<float>& cluster_var_list, 
                                 FlatMatrix<uint8_t>& var_score_list, 
                                 const int qc_num,
                                 float alpha, float beta);


/**
 * @brief 计算每个查询向量到所有质心的距离，找到最近的若干个质心，用于后续分发给 DPU 查询。
 *        通过 qc_num 统一控制本批次的查询类数量，允许比 MAX_QUERY_CLUSTER 小，
 *        同时使用交集查询法决定的 cluster 起始位置。
 * @param query_vectors 查询向量 
 * @param centroids 质心向量 
 * @param query_clusters 返回的每个向量待查询的 clusters 列表 
 * @param split_indices 返回的交集查询法的起始位置，小于该位置则查询 cluster 中全部向量 
 * @param qc_num 本批次所有查询共用的聚类数量，不超过 MAX_QUERY_CLUSTER
 */
void find_query_clusters(const FlatMatrix<DATA_TYPE>& query_vectors, 
                         const FlatMatrix<float>& centroids, 
                         FlatMatrix<int>& query_clusters, 
                         std::vector<uint8_t>& split_indices, 
                         const int qc_num);

/**
 * @brief 计算每个查询向量到所有质心的距离，找到最近的 qc_num 个质心，后续分发给 DPU 查询。
 * @param query_vectors 查询向量 
 * @param centroids 质心向量 
 * @param query_clusters 返回的每个向量待查询的 clusters 列表 
 * @param qc_num 本批次所有查询共用的聚类数量，不超过 MAX_QUERY_CLUSTER
 */
void find_query_clusters(const FlatMatrix<DATA_TYPE>& query_vectors, 
                         const FlatMatrix<float>& centroids, 
                         FlatMatrix<int>& query_clusters,
                         const int qc_num);


/**
 * @brief 对一个批次的原始查询结果进行重排和去重，提取TopK，并存入全局结果矩阵。
 * @param batch_results 当前批次的原始结果 (从DPU pull而来, 展平矩阵).
 * @param topk_results 存放所有查询最终结果的全局矩阵
 * @param batch_size 当前批次中的查询数量
 * @param query_offset 当前批次在所有查询中的全局起始偏移
 */
void rerank_and_store_topk(const FlatMatrix<std::pair<uint32_t, int>>& batch_results,
                           FlatMatrix<int>& topk_results,
                           const uint16_t batch_size,
                           const int query_offset);

/**
 * @brief 将直接从 DPU 一侧 pull 回到缓冲区的结果进行处理和 rerank。
 * @param dpu_stats_info DPU 相关的统计信息
 * @param topk_results 存放所有查询最终结果的全局矩阵
 * @param completed_batch 完成的 batch 智能指针
 */
void process_results_and_rerank(DpuStats& dpu_stats_info, 
                                FlatMatrix<int>& topk_results, 
                                std::shared_ptr<Batch> completed_batch);


/**
 * @brief 将要广播的查询数据存储到 Batch 的 buffer中。
 * @param batch_to_prepare 需要发送的 batch 智能指针
 */
void prepare_broadcast_buffer(std::shared_ptr<Batch> batch_to_prepare);


/**
 * @brief 从每个查询结果中筛选 topK 最小距离的向量，并计算 batch 和 global 准确率。
 * @param ground_truth 真实 topK 的 vector id
 * @param results 查询返回的 distance 和 vector id pair 
 * @param batch_size 查询批次大小
 * @param global_accuracy 全局查询准确率
 */
void rerank_and_calc_accuracy(const FlatMatrix<int32_t>& ground_truth, 
                              const FlatMatrix<std::pair<uint32_t, int>>& results, 
                              const uint16_t batch_size, double& global_accuracy);


/**
 * @brief 根据查询结果，计算平均准确率。
 * @param ground_truth 真实 topK 的 vector id
 * @param topk_results 查询返回的 vector id
 * @param start_idx 计算的开始查询 id
 * @param calc_size 需要计算的梳理
 * @param is_total 是否计算全局查询准确率
 */
void calculate_accuracy(const FlatMatrix<int32_t>& ground_truth,
                        FlatMatrix<int>& topk_results,
                        const int start_idx, const uint16_t calc_size, 
                        const bool is_total);


// 对于当前批次的query，每个query都预先查询raw data，获得上限距离作为hint，然后再开始后续的正式查询
// 后续可以把hint这个环节交给DPU来做，DPU在查询的过程中，查完若干个以后通过回调函数通知host
// 然后host再给DPU发送新的值，这样就不需要发两次query batch了
// 这样的好处，默认我们的数据放在DPU中而host不存原始数据，另hint的操作看起来更有设计
// 后来测了一下，这样的效果提高的不是很多，暂时不做
// void pre_hinted_query(const std::vector<std::vector<DATA_TYPE>>& query_vectors,
//                       const std::vector<std::vector<int>>& query_cluster_ids, 
//                       const std::vector<std::vector<DATA_TYPE>>& base_vectors,
//                       const std::vector<std::vector<float>>& cluster_var_list, 
//                       const std::vector<uint32_t>& top100_max_dist, 
//                       std::vector<uint32_t>& expected_dist_list, 
//                       const int query_offset, const uint16_t batch_size);



#endif // HOST_BASE_HH
