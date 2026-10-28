#include "dpu_batch_kernel.h"


BARRIER_INIT(stats_begin, NR_TASKLETS);
BARRIER_INIT(query_begin, NR_TASKLETS);
MUTEX_INIT(query_mutex);


void process_query(PushPackageHeader *package_header)
{
    perfcounter_t initial_time = perfcounter_config(COUNT_CYCLES, false);
    uint16_t num_queries = package_header->num_vectors;
    uint16_t tid = me();
    thread_max_dist[tid] = CEILING_VALUE; // 先初始化一个略大的上限
    thread_query_num[tid] = 0;  // 重置统计量，让每个tid重置自己的，这个必须重置，其他几个无所谓
    thread_calc_time[tid] = 0;
    // thread_heap_time[tid] = 0;
    // thread_cross_time[tid] = 0;
    thread_end_time[tid] = 0;

    seqreader_buffer_t id_cache = seqread_alloc();
    seqreader_t id_reader;
    seqread_init(id_cache, (__mram_ptr void *)&all_vector_info[0], &id_reader); 

    const uint32_t query_vector_step = sizeof(QueryPackageBody) * NR_TASKLETS;
    const uint32_t result_write_step = sizeof(OneQueryResult) * NR_TASKLETS;
    __mram_ptr uint8_t* my_vec_ptr = recv_buffer + sizeof(PushPackageHeader) + tid * sizeof(QueryPackageBody);
    __mram_ptr uint8_t* my_result_ptr = ret_buffer + sizeof(StatsInfo) + tid * sizeof(OneQueryResult);

    DATA_TYPE curr_vector[ALIGN_VECTOR_DIM];
    for (uint16_t i = tid; i < num_queries; i += NR_TASKLETS) {
        QueryPackageBody query_data;

        // 读取 query 向量
        mram_read((__mram_ptr void *)my_vec_ptr, &query_data, sizeof(QueryPackageBody));
        my_vec_ptr += query_vector_step;
        
        MaxHeap heap = { .size = 0 };  // 初始化最大堆

        // cluster 重叠查询法
        uint8_t qc_num = query_data.qc_num;
        uint8_t cross_cidx_begin = query_data.split_index;  // 交叉查询的cluster的起始id，小于该id的cluster查询全部节点
        // 对于前面的cluster，查询全部节点        
        for (uint8_t cidx = 0; cidx < cross_cidx_begin; ++cidx) {
            process_cluster_range(cidx, &query_data, &id_reader, curr_vector, &heap, false);
        } 
        // 对于后面的cluster，只查询交集节点
        // uint64_t cross_time_begin = perfcounter_get();
        for (uint8_t cidx = cross_cidx_begin; cidx < qc_num - 1; ++cidx) {
            process_cluster_range(cidx, &query_data, &id_reader, curr_vector, &heap, IS_CROSS_QUERY);
        } 
        // uint64_t cross_time_end = perfcounter_get();
        // thread_cross_time[tid] += cross_time_end - cross_time_begin;

        // 写回当下这个查询的结果
        write_query_result((__mram_ptr void *)my_result_ptr, query_data.query_id, &heap);
        my_result_ptr += result_write_step;

        thread_max_dist[tid] = CEILING_VALUE;  // 恢复期望的max_dist到默认值
    }
    thread_end_time[tid] = perfcounter_get() - initial_time;

    barrier_wait(&stats_begin);
    write_stats(initial_time);
}


void process_query_chunk(PushPackageHeader *package_header) 
{
    perfcounter_t initial_time = perfcounter_config(COUNT_CYCLES, false);
    uint32_t num_queries = package_header->num_vectors;
    uint16_t tid = me();
    thread_query_num[tid] = 0;  // 重置统计量，让每个tid重置自己的
    thread_calc_time[tid] = 0;
    // thread_heap_time[tid] = 0;
    // thread_cross_time[tid] = 0;
    thread_end_time[tid] = 0;
    if (tid == 0) global_query_index = 0;  // 重置 global_query_index
    barrier_wait(&query_begin);

    seqreader_buffer_t id_cache = seqread_alloc();
    seqreader_t id_reader;
    seqread_init(id_cache, (__mram_ptr void *)&all_vector_info[0], &id_reader); 

    __mram_ptr uint8_t* query_base_ptr = recv_buffer + sizeof(PushPackageHeader);
    __mram_ptr uint8_t* result_base_ptr = ret_buffer + sizeof(StatsInfo);

    DATA_TYPE curr_vector[ALIGN_VECTOR_DIM];

    uint16_t begin_index, end_index;
    while (get_next_query_chunk(&begin_index, &end_index, num_queries)) {
        // 动态分配 WORK_SIZE 个查询任务
        // __mram_ptr uint8_t* my_vec_ptr = query_base_ptr + begin_index * sizeof(QueryPackageBody);
        // __mram_ptr uint8_t* my_result_ptr = result_base_ptr + begin_index * sizeof(OneQueryResult);
        for (uint16_t i = begin_index; i < end_index; ++i) {
            QueryPackageBody query_data;

            __mram_ptr uint8_t* my_vec_ptr = query_base_ptr + i * sizeof(QueryPackageBody);
            __mram_ptr uint8_t* my_result_ptr = result_base_ptr + i * sizeof(OneQueryResult);

            // 读取 query 向量
            mram_read((__mram_ptr void *)my_vec_ptr, &query_data, sizeof(QueryPackageBody));
            my_vec_ptr += sizeof(QueryPackageBody);

            MaxHeap heap = { .size = 0 };

            uint8_t cross_cidx_begin = query_data.split_index;
            uint8_t qc_num = query_data.qc_num;
            for (uint8_t cidx = 0; cidx < cross_cidx_begin; ++cidx) {
                process_cluster_range(cidx, &query_data, &id_reader, curr_vector, &heap, false);
            }
            for (uint8_t cidx = cross_cidx_begin; cidx < qc_num - 1; ++cidx) {
                process_cluster_range(cidx, &query_data, &id_reader, curr_vector, &heap, IS_CROSS_QUERY);
            }

            // 写回结果
            write_query_result((__mram_ptr void *)my_result_ptr, query_data.query_id, &heap);
            my_result_ptr += sizeof(OneQueryResult);
        }
    }
    thread_end_time[tid] = perfcounter_get() - initial_time;

    barrier_wait(&stats_begin);
    write_stats(initial_time);
}


bool get_next_query_chunk(uint16_t *start, uint16_t *end, uint32_t num_queries) 
{
    mutex_lock(query_mutex);
    uint16_t curr = global_query_index;
    global_query_index += WORK_SIZE;
    mutex_unlock(query_mutex);

    if (curr >= num_queries) return 0;  // 已经没有了
    *start = curr;
    *end = (curr + WORK_SIZE > num_queries) ? num_queries : curr + WORK_SIZE;
    return 1;
}
