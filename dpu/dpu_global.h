#ifndef DPU_GLOBAL_H
#define DPU_GLOBAL_H

#include <mram.h>
#include <stdint.h>
#include <stdbool.h>
#include <perfcounter.h>
#include "../share/global.h"

#define ID_CACHE_SIZE 20  // 存储数据阶段 ID 在 WRAM 中的缓存大小，必须是4的倍数
#define WORK_SIZE 6  // 每一个线程每次取的查询量


// 定义是否采用交集查询，baseline 不采用交集查询，则不开启距离计算优化
// 定义 L2 距离计算的方案，0 为默认计算，1 为考虑 segment 方差顺序计算，2 为 segment + 阈值早退
#if (IS_BASELINE == true)
    #define IS_CROSS_QUERY false
    #define L2_DIST_METHOD 0
#else
    #define IS_CROSS_QUERY true
    #define L2_DIST_METHOD 1
#endif

// top100最大的可能上限，这个即使设置大了，也慢不了多少，因为也就多算了几个堆没有满时的元素
#define CEILING_VALUE (1 << 30)


// 一个向量自己的 vid 和所属的 cids 记录结构体
typedef struct {
    int vid; 
    #if (MAX_REPLECA <= 2)
        // 只需要记录单个 cross_cid 
        #define USING_SIMPLE_CID true
        uint16_t cross_cid;
    #else 
        // 使用 VectorClusterMap 结构体存储所有向量各自所属的cid列表
        #define USING_SIMPLE_CID false
        VectorClusterMap cross_cid;
    #endif
} IdInfo;


// 单个cluster元信息存储（不含向量数据）
typedef struct {
    int offset;       // 向量在大数组中的起始位置
    int num_vectors;  // cluster中向量数量
} ClusterHeader;

// 这里前缀只需要使用 extern 关键字，不需要包含其他与存储类型或属性相关的前缀
// 因为 extern 声明仅用于告诉编译器该变量在其他地方定义，而不涉及变量的实际存储类型或属性

extern uint32_t thread_max_dist[NR_TASKLETS];   // 每个线程负责的当前查询中，期望返回的距离最大值
extern uint64_t thread_query_num[NR_TASKLETS];  // 记录每一个线程的计算量

extern perfcounter_t thread_heap_time[NR_TASKLETS];  // 堆操作的时间
extern perfcounter_t thread_calc_time[NR_TASKLETS];  // DPU计算距离的时间 
extern perfcounter_t thread_cross_time[NR_TASKLETS]; // 交集判断的时间
extern perfcounter_t thread_end_time[NR_TASKLETS];   // 每个线程查询结束的时间

extern uint32_t global_query_index;  // 采用每个线程每次取WORK_SIZE个查询时使用的全局共享变量

extern uint32_t global_dpu_batch_id;  // 记录查询过程中 DPU 一共启动多少次，用于时间统计
extern __mram_noinit uint64_t dpu_time_stats[MAX_DPU_TIME_STAMP]; // 记录dpu到底在哪些时间干活的时间对


/* 下面是采用定义MRAM中数组的写法，由DPU内部的代码来决定数组分配的位置 */

// 数据传输缓冲区
extern __mram_noinit uint8_t recv_buffer[MAX_PUSH_BUF_SIZE];  // 接收query/raw data的缓冲区
extern __mram_noinit uint8_t ret_buffer[MAX_PULL_BUF_SIZE];   // 返回查询结果的缓冲区

// 所有向量统一存储，因为跟id共用一个offset，所以也要加一个缓冲
extern __mram_noinit DATA_TYPE all_vectors[MAX_VECTORS_PER_DPU + (NUM_CLUSTER << 2)][ALIGN_VECTOR_DIM];

// 所有的向量ID和cross_id一起存储，这里需要加一个缓冲的数量
extern __mram_noinit IdInfo all_vector_info[MAX_VECTORS_PER_DPU + (NUM_CLUSTER << 2)]; 

// 所有cluster的头信息
extern __mram_noinit ClusterHeader cluster_headers[NUM_CLUSTER];



#endif // DPU_GLOBAL_H
