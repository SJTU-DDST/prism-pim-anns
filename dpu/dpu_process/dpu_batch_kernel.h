#ifndef DPU_BATCH_KERNEL_H
#define DPU_BATCH_KERNEL_H

#include "dpu_base.h"


// DPU batch process 模式的查询函数，默认使用 cluster 交集法，支持多线程，每个线程取 input 中对应位置的查询
void process_query(PushPackageHeader *package_header);

// DPU batch process 模式的查询函数，默认使用 cluster 交集法，支持多线程，每个线程取 input 中的一个 chuck 处理
void process_query_chunk(PushPackageHeader *package_header);

// 每个线程获取下一个要查询的 chuck，仅 process_query_chunk 使用
bool get_next_query_chunk(uint16_t *start, uint16_t *end, uint32_t num_queries);

#endif // DPU_BATCH_KERNEL_H