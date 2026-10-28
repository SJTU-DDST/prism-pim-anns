#ifndef SEND_TO_DPU_HH
#define SEND_TO_DPU_HH

#include <dpu> // 确保host中的UPMEM的dpu库不会被多次include，目前只有本文件include
#include <vector>
#include <queue>
#include <unordered_set>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <algorithm>

#include "../utils.hh"
#include "../stats.hh"
#include "../data_utils/flat_matrix.hh" 


/**
 * @brief 将原始数据发送给 DPU。具体方式为遍历每一个 cluster，可选用不同方式分发数据。
 * @param system DPU 系统
 * @param vectors 待分发的原始数据
 * @param clusters 聚类质心id，DPU 端需要记录原始数据在聚类中所属的类别 id
 * @param multi_labels 多聚类标签，标记每一个节点所属的类别
 */
void send_raw_data(dpu::DpuSet& system,
                   const FlatMatrix<DATA_TYPE>& vectors,
                   const JaggedArray<int>& clusters,
                   const JaggedArray<int>& multi_labels);

/**
 * @brief 把 cluster 中的数据平均切成若干个 block 分发给所有 DPU。
 * @param system DPU 系统
 * @param vectors 待分发的原始数据
 * @param vids_in_cluster 一个 cluster 包含的 vector_id
 * @param multi_labels 多聚类标签，标记每一个节点所属的类别
 * @param curr_cid 当前的 cluster_id
 */
void dispatch_cluster_in_block(dpu::DpuSet& system,
                               const FlatMatrix<DATA_TYPE>& vectors,
                               std::span<const int> vids_in_cluster, 
                               const JaggedArray<int>& multi_labels, 
                               const int curr_cid);

/**
 * @brief 以 hash 方式将每一个 cluster 的数据平均分发到所有 DPU。
 *        可选按照 hash 函数的方式随机分发，也可选按照 mod NR_DPUS 的方式平均分发。
 * @param system DPU 系统
 * @param vectors 待分发的原始数据
 * @param vids_in_cluster 一个 cluster 包含的 vector_id
 * @param multi_labels 多聚类标签，标记每一个节点所属的类别
 * @param curr_cid 当前的 cluster_id
 */
void dispatch_cluster_in_hash(dpu::DpuSet& system,
                              const FlatMatrix<DATA_TYPE>& vectors,
                              std::span<const int> vids_in_cluster, 
                              const JaggedArray<int>& multi_labels, 
                              const int curr_cid);


/**
 * @brief 以广播方式将一个 batch 的查询向量分发到所有 DPU。
 * @param system DPU 系统
 * @param query_vectors 待广播的查询向量
 * @param query_cids 查询向量需要查询的 cluster_id 列表
 * @param split_indices 决定使用全部查询和交集查询的 index 分界
 * @param qc_num 本批次所有查询共用的 cluster 数量
 * @param var_score_list 向量分段后的方差列表，按照从高到低排序 
 * @param query_offset 查询批次的偏移 
 */
void broadcast_query_batch(dpu::DpuSet& system, 
                           const FlatMatrix<DATA_TYPE>& query_vectors,
                           const FlatMatrix<int>& query_cids, 
                           const std::vector<uint8_t>& split_indices, 
                           const int qc_num,
                           const FlatMatrix<uint8_t>& var_score_list, 
                           const int query_offset);

/**
 * @brief 从 DPU 中收回一个 batch 的查询结果。
 * @param system DPU 系统
 * @param results 写入查询结果的结构，每一个查询结果保存 <dist, id>，其中 dist 用于后续重排序
 * @param dpu_stats_info 保存 DPU 的统计信息
 * @param query_offset 查询批次的偏移 
 * @param batch_size 查询批次的大小 
 */
void pull_query_result(dpu::DpuSet& system, 
                       FlatMatrix<std::pair<uint32_t, int>>& results, 
                       DpuStats& dpu_stats_info, 
                       const int query_offset, 
                       const uint16_t batch_size);

/**
 * @brief 从 DPU 中收回一个 batch 的查询结果，并提取 topk。
 * @param system DPU 系统
 * @param topk_results 保存全部查询结果的 TopK 
 * @param dpu_stats_info 保存 DPU 的统计信息
 * @param query_offset 查询批次的偏移 
 * @param batch_size 查询批次的大小 
 */
void pull_results_and_extract_topk(dpu::DpuSet& system, 
                                   FlatMatrix<int>& topk_results,
                                   DpuStats& dpu_stats_info, 
                                   const int query_offset, 
                                   const uint16_t batch_size);

/**
 * @brief 最简单的从 dpu 一侧拉回结果的函数，只拷贝到结果缓冲区，不做任何处理。
 * @param system DPU 系统
 * @param pull_buffers 保存结果的缓冲区
 */
void simple_pull_results(dpu::DpuSet& system, std::vector<std::vector<uint8_t>>& pull_buffers);


/**
 * @brief 最简单的给 dpu 广播查询数据并执行的函数，只将数据从缓冲区中广播出去并运行 DPU。
 * @param system DPU 系统
 * @param dpu_stats_info DPU 一侧的计时器，统计 DPU 真正开始运行的时间
 * @param pull_buffers 保存结果的缓冲区
 */
void simple_broadcast_query(dpu::DpuSet& system, DpuStats& dpu_stats_info, 
                            const std::chrono::steady_clock::time_point& global_start, 
                            std::vector<uint8_t>& broadcast_buffer);


/**
 * @brief 从所有 DPU 拉取 batch 计时数据
 * @param system DpuSet 对象
 * @param pull_buffers 保存 batch time stamp 的缓冲区
 */
void pull_batch_time_stamp(dpu::DpuSet& system, std::vector<std::vector<uint64_t>>& time_buffers);


/**
 * @brief 获取 DPU 每秒有多少个 clock cycle，用于后续计算 DPU 内部各模块的运行时间。
 *        这是一个常数，但模拟器和实机不一样，所以需要get一下。
 * @param system DPU 系统
 */
static uint32_t get_dpu_clock(dpu::DpuSet& system)
{
    auto dpu = system.dpus()[0];  // 注意 dpus() 方法在异步系统中不能用
    std::vector<std::vector<uint32_t>> clocks_per_sec(1, std::vector<uint32_t>(1, 0));
    dpu->copy(clocks_per_sec, "CLOCKS_PER_SEC");  // 获取常数
    return clocks_per_sec.front().front();
}


/* 以下函数暂时用不到 */

/**
 * @brief 以广播方式将一个 PQ 得到的码本分发给所有 DPU。
 * @param system DPU 系统
 * @param pq_codebook PQ 得到的码本 
 */
void broadcast_pq_codebook(dpu::DpuSet& system, 
                           const std::vector<std::vector<std::vector<DATA_TYPE>>>& pq_codebook);


/**
 * @brief 将 PQ 编码后的数据发送给 DPU。具体方式为遍历每一个 cluster，分批发送。
 * @param system DPU 系统
 * @param pq_codes 编码后的向量
 * @param cluster_ids 聚类质心id，DPU 端需要记录原始数据在聚类中所属的类别 id
 * @param cluster_to_dpus cluster 划分到的 DPU
 */
void send_pq_codes(dpu::DpuSet& system,
                   const std::vector<std::vector<uint8_t>>& pq_codes,
                   const std::vector<std::vector<int>>& cluster_ids,
                   const std::vector<std::vector<int>>& cluster_to_dpus);


#endif // SEND_TO_DPU_HH
