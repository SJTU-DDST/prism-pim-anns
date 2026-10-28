#include "dpu_global.h"


// 注意所有全局变量只能定义一次，如果定义在头文件中，被多次include，则会被定义多次
// 因此需要定义在.c文件中，而.h文件使用 `extern` 进行声明


// ------------------------------------------------------------------------------------
// 采用定义MRAM中数组的写法，由DPU内部的代码来决定数组分配的位置
// ------------------------------------------------------------------------------------
// 数据传输缓冲区，放在 MRAM 中
__mram_noinit uint8_t recv_buffer[MAX_PUSH_BUF_SIZE];  // 接收query data的缓冲区
__mram_noinit uint8_t ret_buffer[MAX_PULL_BUF_SIZE];   // 返回查询结果的缓冲区

// 所有的向量统一存储
__mram_noinit DATA_TYPE all_vectors[MAX_VECTORS_PER_DPU + (NUM_CLUSTER << 2)][ALIGN_VECTOR_DIM];

// 所有的向量ID和cross_id一起存储，这里需要加一个缓冲的数量
__mram_noinit IdInfo all_vector_info[MAX_VECTORS_PER_DPU + (NUM_CLUSTER << 2)]; 

// 所有cluster的头信息
__mram_noinit ClusterHeader cluster_headers[NUM_CLUSTER];


uint32_t thread_max_dist[NR_TASKLETS];   // 每个线程负责的当前查询中，期望返回的距离最大值
uint64_t thread_query_num[NR_TASKLETS] = {0};  // 记录每一个线程的计算量

perfcounter_t thread_heap_time[NR_TASKLETS] = {0};  // 堆操作的时间
perfcounter_t thread_calc_time[NR_TASKLETS] = {0};  // DPU计算距离的时间 
perfcounter_t thread_cross_time[NR_TASKLETS] = {0}; // 交集判断的时间
perfcounter_t thread_end_time[NR_TASKLETS] = {0};   // 每个线程查询结束的时间

uint32_t global_query_index = 0;  // 采用每个线程每次取WORK_SIZE个查询时使用的全局共享变量

uint32_t global_dpu_batch_id = 0;  // 记录查询过程中 DPU 一共启动多少次，用于时间统计
__mram_noinit uint64_t dpu_time_stats[MAX_DPU_TIME_STAMP]; // 记录dpu到底在哪些时间干活的时间对


