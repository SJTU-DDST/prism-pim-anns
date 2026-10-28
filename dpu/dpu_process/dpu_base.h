#ifndef DPU_BASE_HH
#define DPU_BASE_HH

#include <seqread.h>
#include <mutex.h>
#include <assert.h>
#include <barrier.h>
#include <defs.h> // include 获取当前线程 ID 的 me() 函数
#include <alloc.h>

#include "../heap.h"
#include "../dpu_utils.h"
#include "../dpu_global.h"


/**
 * @brief 用于 batch 模式下计时的结构体
 */
typedef struct CounterCycles{
    perfcounter_t start;
    perfcounter_t end;
} CounterCycles;


/**
 * @brief 单线程将接收缓冲区中的向量数据存储到 MRAM
 * @param package_header 指向MRAM中包头的指针
 */
void store_raw_vectors(PushPackageHeader *package_header);


/**
 * @brief 多线程将接收缓冲区中的向量数据存储到 MRAM
 * @note 此函数由所有 Tasklet 并行执行。
 * @param package_header 指向MRAM中包头的指针
 */
void store_raw_vectors_mt(PushPackageHeader *package_header);

/**
 * @brief 处理若干个 cluster 范围内的查询函数
 * @note 此函数由所有Tasklet并行执行。
 * @param cidx 当前查询的 cluster_index 
 * @param query_data 查询数据结构体 (含待查询的 query vector 和 meta data)
 * @param id_reader 用于成块读取 vector_id 的结构体
 * @param curr_vector 存储读取的 vector 的向量内存空间
 * @param heap 用于距离排序的堆
 * @param use_filter 是否采用交集过滤法
 */
void process_cluster_range(const uint8_t cidx, QueryPackageBody* query_data, seqreader_t *id_reader, 
                           DATA_TYPE *curr_vector, MaxHeap *heap, const bool use_filter);

/**
 * @brief 处理单个查询向量的函数。
 *        可采用启发式计算，每次先算所有 segment 中差异最大的，结合阈值过滤，尽早踢出无效结果
 * @note 此函数由所有Tasklet并行执行。
 * @param vec_offset 向量在 DPU 中存储的偏移
 * @param id 向量的全局 id
 * @param query_data 查询数据结构体 (含待查询的 query vector 和 meta data)
 * @param curr_vector 存储读取的 vector 的向量内存空间
 * @param heap 用于距离排序的堆
 * @param cidx 当前查询的 cluster_index 
 */
void process_single_vector(const int vec_offset, int id, QueryPackageBody* query_data, 
                           DATA_TYPE *curr_vector, MaxHeap *heap, const uint8_t cidx);


/**
 * @brief 判断当前 raw data vector 是否属于查询范围中两个 cluster 的交集。
 *        采用双指针方案，只在 MAX_REPLICA >= 3 时启用。
 * @note 此函数由所有Tasklet并行执行。
 * @param vec_cluster_ptr 指向该向量多标签存储的指针
 * @param cluster_ids 向量原始所属的 cluster_id
 * @param qc_num 查询的 cluster 数量
 * @param cidx 当前查询的 cluster_index 
 */
// bool is_vec_in_cross_cluster(const VectorClusterMap *vec_cluster_ptr, uint16_t* cluster_ids, 
//                              const uint8_t qc_num, const uint8_t cidx);


/**
 * @brief 从最大堆中写回一个 query vector 的查询结果，仅 DPU batch process 模式下使用
 * @param my_result_ptr Tasklet 自己写结果的指针
 * @param query_id 查询 id
 * @param heap 用于距离排序的堆
 */
void write_query_result(__mram_ptr void *my_result_ptr, const uint32_t query_id, MaxHeap *heap);


/**
 * @brief 处理并写回 DPU 的统计信息
 * @param batch_initial_time 当前 batch 开始的时间
 */
void write_stats(perfcounter_t batch_initial_time);


#endif // DPU_BASE_HH

