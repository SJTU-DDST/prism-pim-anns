#include "dpu_base.h"


static int total_vector_num;  // 所有 cluster 共用的全局 offset，store raw data使用

BARRIER_INIT(store_data, NR_TASKLETS);

void store_raw_vectors(PushPackageHeader *package_header) 
{
    int cluster_id = package_header->data_cluster;
    int num_vectors = package_header->num_vectors;  // 后面要用到负数，所以用 int

    ClusterHeader header;
    if (package_header->is_new == 1) {
        // 新的 cluster
        header.offset = total_vector_num;
        header.num_vectors = num_vectors;
    } else {
        // 仍然是上一个cluster的数据，因此读取原有的头部，继续在原有的cluster后面添加数据
        mram_read((__mram_ptr void *)&cluster_headers[cluster_id], &header, sizeof(ClusterHeader));        
        header.num_vectors += num_vectors; // head.offset不变，只增加num_vectors
    }
    // 写回 cluster header
    mram_write(&header, (__mram_ptr void *)&cluster_headers[cluster_id], sizeof(ClusterHeader));

    __mram_ptr uint8_t* vec_ptr = recv_buffer + sizeof(PushPackageHeader); // 这里不进行计算，直接按照uint8_t读取

    // ===== 向量相关 ID 聚合写入 =====
    IdInfo id_buffer[ID_CACHE_SIZE];
    int buf_idx = 0; // buffer 索引
    int vid_offset = total_vector_num;

    for (int i = 0; i < num_vectors; ++i) {
        PushPackageBody data_body;
        mram_read((__mram_ptr void *)vec_ptr, &data_body, sizeof(PushPackageBody)); // 读取向量
        vec_ptr += sizeof(PushPackageBody);

        // 写向量本体
        mram_write(data_body.vector, (__mram_ptr void *)&all_vectors[total_vector_num + i], ALIGN_VECTOR_DIM); 
        id_buffer[buf_idx] = (IdInfo){data_body.vector_id, data_body.cross_cid};
        buf_idx++;

        // 缓存满 ID_CACHE_SIZE 个，flush 到 MRAM
        if (buf_idx == ID_CACHE_SIZE) {
            mram_write(id_buffer, (__mram_ptr void *)&all_vector_info[vid_offset], ID_CACHE_SIZE * sizeof(IdInfo)); 
            vid_offset += ID_CACHE_SIZE;
            buf_idx = 0;
        }
    }

    int needs_padding = (-num_vectors) & 0x3; // 填充到 4 的倍数
    // 如果还有剩余未写满的 ID，补齐 padding 并写入
    if (buf_idx > 0) {
        int padded_id_count = buf_idx + needs_padding; // 计算实际写入的字节数
        mram_write(id_buffer, (__mram_ptr void *)&all_vector_info[vid_offset], padded_id_count * sizeof(IdInfo));
    }

    // 注意继续写的时候，因为前面定义的DATA_BATCH_SIZE（也就是这里的num_vectors）强制是一个偶数，因此一定是对齐的
    total_vector_num = total_vector_num + num_vectors + needs_padding;  // 更新全局 offset，包括 padding
    assert (total_vector_num <= MAX_VECTORS_PER_DPU);
}


void store_raw_vectors_mt(PushPackageHeader *package_header) 
{
    uint16_t tid = me();
    int num_vectors = package_header->num_vectors;

    // 阶段 1: Tasklet 0 处理元数据，其他Tasklet等待
    if (tid == 0) {
        int cluster_id = package_header->data_cluster;
        ClusterHeader header;
        if (package_header->is_new == 1) {
            // 新的 cluster
            header.offset = total_vector_num;  // 写入全局变量
            header.num_vectors = num_vectors;
        } else {
            // 追加到现有cluster，读取旧头部
            mram_read((__mram_ptr void *)&cluster_headers[cluster_id], &header, sizeof(ClusterHeader));
            header.num_vectors += num_vectors;  // 更新向量数量，offset保持不变
        }
        // 写回更新后的 cluster header
        mram_write(&header, (__mram_ptr void *)&cluster_headers[cluster_id], sizeof(ClusterHeader));
    }

    // 等待Tasklet 0完成上述操作，并确保所有tasklet都能读取到正确的total_vector_num和wram_num_vectors
    barrier_wait(&store_data);

    // 阶段 2: 所有Tasklet并行处理数据拷贝    
    // 从共享变量中读取本次任务的全局信息
    uint32_t start_offset = total_vector_num;

    // 为了保证MRAM 8字节对齐写入，我们将向量按4个一组进行划分(chunk)，并将这些组分配给tasklet。
    // 这样可以确保每个tasklet（除了最后一个）处理的向量数是4的倍数。
    int my_start_idx = 0;
    int my_num_vectors = 0;

    // 1. 计算可以被4整除的 "块" 的数量，以及余下的向量数
    int num_chunks_of_4 = num_vectors / 4;
    int remaining_vectors = num_vectors % 4;

    // 2. 将这些 "块" 均匀分配给所有tasklet
    int chunks_per_tasklet = num_chunks_of_4 / NR_TASKLETS;
    int remainder_chunks = num_chunks_of_4 % NR_TASKLETS;

    int my_chunks = chunks_per_tasklet + (tid < remainder_chunks ? 1 : 0);
    my_num_vectors = my_chunks * 4;

    // 3. 根据分配到的块，计算每个tasklet的起始向量索引
    int prev_chunks = tid * chunks_per_tasklet + (tid < remainder_chunks ? tid : remainder_chunks);
    my_start_idx = prev_chunks * 4;

    // 4. 将剩余的向量(0-3个)全部交给最后一个tasklet处理
    if (tid == NR_TASKLETS - 1) {
        my_num_vectors += remaining_vectors;
    }

    int padded_count = 0; // 是否需要对齐来补全 8 字节

    if (my_num_vectors > 0) {
        // 每个tasklet拥有独立的WRAM缓冲区，避免冲突
        IdInfo id_buffer[ID_CACHE_SIZE];
        int buf_idx = 0;

        // 计算当前tasklet的读写起始地址
        __mram_ptr uint8_t* my_vec_ptr = recv_buffer + sizeof(PushPackageHeader) + my_start_idx * sizeof(PushPackageBody);
        int my_vid_offset = start_offset + my_start_idx;
        int my_vid_cache_offset = my_vid_offset;

        for (int i = 0; i < my_num_vectors; ++i) {
            PushPackageBody data_body;
            // 1. 读取一个向量的完整数据 (ID + vector)
            mram_read(my_vec_ptr, &data_body, sizeof(PushPackageBody));
            my_vec_ptr += sizeof(PushPackageBody);

            // 2. 将向量本体写入MRAM (本身是128字节对齐的)
            mram_write(data_body.vector, (__mram_ptr void *)&all_vectors[my_vid_offset + i], ALIGN_VECTOR_DIM);
            
            // 3. 将ID写入WRAM缓冲区
            id_buffer[buf_idx] = (IdInfo){data_body.vector_id, data_body.cross_cid};
            buf_idx++;

            // 4. 当WRAM缓冲区满了，批量写回MRAM
            if (buf_idx == ID_CACHE_SIZE) {
                mram_write(id_buffer, (__mram_ptr void *)&all_vector_info[my_vid_cache_offset], ID_CACHE_SIZE * sizeof(IdInfo));
                my_vid_cache_offset += ID_CACHE_SIZE;
                buf_idx = 0;
            }
        }

        // 处理循环结束后缓冲区中剩余的ID
        if (buf_idx > 0) {
            // 为保证MRAM写入尺寸是8字节的倍数，需要将待写入的元素数量向上补齐到4的倍数
            padded_count = (buf_idx + 3) & ~0x3; // ~0x3 是 ...11111100

            // 用0填充WRAM缓冲区中多出来的位置，防止写入垃圾数据
            for (int k = buf_idx; k < padded_count; ++k) {
                id_buffer[k] = (IdInfo){0, 0};
            }

            // 使用补齐后的数量进行写入，确保写入尺寸是对齐的
            mram_write(id_buffer, (__mram_ptr void *)&all_vector_info[my_vid_cache_offset], padded_count * sizeof(IdInfo));
        }
    }

    // 等待所有tasklet完成数据拷贝
    barrier_wait(&store_data);

    // 阶段 3: Tasklet 0 负责最后的全局状态更新
    if (tid == 0) {
        // 计算需要padding的数量，以确保下一次写入是4字节对齐的
        int needs_padding = (-num_vectors) & 0x3; // 填充到4的倍数
        total_vector_num = start_offset + num_vectors + needs_padding;  // 更新全局向量总数
        assert (total_vector_num <= MAX_VECTORS_PER_DPU);
    }
}


void process_cluster_range(const uint8_t cidx, QueryPackageBody* query_data, seqreader_t *id_reader, 
                           DATA_TYPE *curr_vector, MaxHeap *heap, const bool use_filter)
{
    uint16_t cluster_id = query_data->cluster_ids[cidx];
    ClusterHeader header;
    mram_read((__mram_ptr void *)&cluster_headers[cluster_id], &header, sizeof(ClusterHeader));

    int offset = header.offset;
    int count = header.num_vectors;

    IdInfo *id_ptr = seqread_seek((__mram_ptr void *)&all_vector_info[offset], id_reader);
    for (int j = 0; j < count; ++j) {
        IdInfo curr_id = *id_ptr;
        id_ptr = seqread_get(id_ptr, sizeof(IdInfo), id_reader);

        #if (NR_DPUS <= 128)
            if (is_in_heap(heap, curr_id.vid)) {
                continue;  // 只在数量少的 DPU 时使用，因为重复的 cid 可能已经在 heap 里面了；DPU 特别多时可忽略
            }
        #endif

        if (use_filter) {
            uint16_t cross_cid = curr_id.cross_cid;

            // 提前判断是否 cross_cid < 交集部分最小值，否则直接下一轮
            if (cross_cid < query_data->cluster_ids[cidx + 1]) {
                continue;  // cluster_ids 是升序排列的, 可以 early stop
            }

            // 遍历匹配法，通过测试发现比二分查找法效果好一点点
            bool is_cross = false;
            for (uint8_t k = cidx + 1; k < query_data->qc_num; ++k) {
                if (cross_cid == query_data->cluster_ids[k]) {
                    is_cross = true;
                    break;
                } 
            }

            if (!is_cross) continue; // 该向量不在两个类的交集中，不查询
        }
        process_single_vector(offset + j, curr_id.vid, query_data, curr_vector, heap, cidx);
        thread_query_num[me()]++;
    }
}


// 对扩展到每个 vector 副本数量到3个或更多个 cluster 中使用，作为对比，说明副本数量为2效果最好
// bool is_vec_in_cross_cluster(const VectorClusterMap *vec_cluster_ptr, uint16_t* cluster_ids,
//                              const uint8_t qc_num, const uint8_t cidx) 
// {
//     uint8_t i = 0;  // VectorClusterMap 中的 cluster id 的偏移
//     uint8_t j = cidx + 1;  // 交集部分开始的 cluster id

//     while (i < vec_cluster_ptr->ccount && j < qc_num) {
//         uint16_t cid = vec_cluster_ptr->cids[i];
//         uint16_t qid = cluster_ids[j];

//         if (cid == qid) {
//             return true;  // 找到交集
//         } else if (cid < qid) {
//             ++i;
//         } else {
//             ++j;
//         }
//     }
//     return false;  // 无交集
// }


inline void process_single_vector(const int vec_offset, int id, QueryPackageBody* query_data, 
                                  DATA_TYPE *curr_vector, MaxHeap *heap, const uint8_t cidx)
{
    mram_read((__mram_ptr void *)&all_vectors[vec_offset], curr_vector, ALIGN_VECTOR_DIM);
    uint64_t calc_start = perfcounter_get();

    #if (L2_DIST_METHOD == 0)
        uint32_t dist = l2_distance_lut((DATA_TYPE*)(query_data->vector), curr_vector);
    #elif (L2_DIST_METHOD == 1)
        uint32_t dist = l2_distance_segment_check((DATA_TYPE*)(query_data->vector), curr_vector, query_data->check_dims[cidx]);
    #else
        #error "Unsupported L2_DIST_METHOD."
    #endif

    uint64_t calc_end = perfcounter_get();
    thread_calc_time[me()] += calc_end - calc_start;

    // uint32_t heap_start = perfcounter_get();
    heap_insert(heap, id, dist);
    // uint32_t heap_end = perfcounter_get();
    // thread_heap_time[me()] += heap_end - heap_start;
}


void write_query_result(__mram_ptr void *my_result_ptr, const uint32_t query_id, MaxHeap *heap)
{
    OneQueryResult query_result;
    query_result.query_id = query_id;
    // query_result.return_num = heap->size;

    // printf("query_id:%d, heap->size:%d\n", query_id, heap->size);
    for (int k = 0; k < MAX_RETURNED_PER_DPU; ++k) {
        query_result.neighbor_ids[k] = heap->nodes[k].index;
        query_result.distance[k] = heap->nodes[k].distance;
        // printf("query_id:%d, heap->size:%d, k:%d, return id:%d, dist:%u\n", 
        //        query_id, heap->size, k, heap->nodes[k].index, heap->nodes[k].distance);
    }
    mram_write(&query_result, my_result_ptr, sizeof(OneQueryResult));
}


void write_stats(perfcounter_t batch_initial_time)
{
    if (me() == 0) {
        perfcounter_t batch_end_time = perfcounter_get();
        uint64_t total_query_num = 0;
        perfcounter_t total_heap_time = 0;
        perfcounter_t total_calc_time = 0;
        perfcounter_t total_cross_time = 0;
        perfcounter_t max_time = thread_end_time[0];  // 初始化线程完成时间的最大值和最小值
        perfcounter_t min_time = thread_end_time[0];
        for (uint16_t i = 0; i < NR_TASKLETS; ++i) {
            // printf("tid:%u, calc:%lu, heap:%lu, cross:%lu (cycles)\n", i, thread_calc_time[i], thread_heap_time[i], thread_cross_time[i]);
            total_query_num += thread_query_num[i];
            total_calc_time += thread_calc_time[i];
            // total_heap_time += thread_heap_time[i];  // 暂不统计 heap 时间
            // total_cross_time += thread_cross_time[i];  // 暂不统计 cross 时间
            if (thread_end_time[i] > max_time) max_time = thread_end_time[i];
            if (thread_end_time[i] < min_time) min_time = thread_end_time[i];
        }
        StatsInfo stats_info = {total_query_num, total_calc_time, total_heap_time, total_cross_time, 
                                max_time - min_time, max_time};
        mram_write(&stats_info, (__mram_ptr void *)ret_buffer, sizeof(StatsInfo));  // StatsInfo 写在 ret_buffer 最前面

        CounterCycles batch_cycles_stats = (CounterCycles){batch_initial_time, batch_end_time};
        // assert(global_dpu_batch_id < (MAX_DPU_TIME_STAMP / 2));
        mram_write(&batch_cycles_stats, (__mram_ptr void *)&dpu_time_stats[global_dpu_batch_id << 1], sizeof(CounterCycles));
        global_dpu_batch_id++;  // 记录查询的批次号
    }
}


