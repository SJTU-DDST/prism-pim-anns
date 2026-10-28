#ifndef GLOBAL_HH
#define GLOBAL_HH

// 本文件存放 host 和 DPU 之间共享的全局变量和结构体定义，遵循 C 风格

// ------------------------------------------------------------------------------------
// 根据数据集名称定义维度、数据类型、分组计算信息 
// ------------------------------------------------------------------------------------
#define DATA_NAME 1  // 0: sift, 1: spacev
#if DATA_NAME == 0
    #define VECTOR_DIM 128  // 向量维度
    #define ALIGN_VECTOR_DIM 128  // 传输和DPU读取中需要8字节对齐的维度
    #define DATA_TYPE uint8_t
    #define BUCKET_SIZE 16  // 检查的步长，每个步长中选取绝对值最大的先开始计算
    // BUCKET_SIZE 8 的效果更好一些（检查点增加到16个）
    // 也尝试了用4bits而非uint8的压缩方案，由于压缩和解压缩耗时，效果不好

#elif DATA_NAME == 1
    #define VECTOR_DIM 100  // 向量维度
    #define ALIGN_VECTOR_DIM 104  // 传输和DPU读取中需要8字节对齐的维度
    #define DATA_TYPE int8_t
    #define BUCKET_SIZE 10  // 检查的步长，每个步长中选取绝对值最大的先开始计算

#else
    #error "DATA_NAME must be 0 (sift) or 1 (spacev)."
#endif


// ------------------------------------------------------------------------------------
// 聚类相关的宏 
// ------------------------------------------------------------------------------------
#define IS_BASELINE false
#if (IS_BASELINE == true)
    // 每个数据点至少复制到多少个其他簇
    #define MIN_REPLICA 1
    // 每个数据点最多可以复制到多少个其他簇
    #define MAX_REPLICA 1
#else
    // 每个数据点至少复制到多少个其他簇
    #define MIN_REPLICA 2
    // 每个数据点最多可以复制到多少个其他簇
    #define MAX_REPLICA 2
#endif

#if !(MAX_REPLICA >= MIN_REPLICA)
    #error "MAX_REPLICA must be >= MIN_REPLICA"
#endif


// ------------------------------------------------------------------------------------
// 分发数据和查询相关的通用宏 
// ------------------------------------------------------------------------------------
// 每个查询向量最终需要的最近邻数量
#define TOP_K 100  
#define MAX_PUSH_BUF_SIZE (1024 * 256)  // push 缓冲区最大预留空间
#define MAX_PULL_BUF_SIZE (1024 * 8)   // host 从 DPU 一次最多收回的字节数
#define MAX_VECTORS_PER_DPU (1024 * 256) // 单个DPU中最大存储的向量数，最大占用48MiB的空间
#define NUM_CHECKPOINTS (VECTOR_DIM / BUCKET_SIZE)  // 检查点个数，用于提前终止向量的距离计算
#define MAX_DPU_TIME_STAMP 1024  // DPU 一侧存储的最大时间戳对数


// ------------------------------------------------------------------------------------
// 根据数据集规模来定义的相关的宏 
// ------------------------------------------------------------------------------------
// 1: 10K, 2: 1M, 3: 10M, 4: 100M
#define DATA_SIZE 4
#if DATA_SIZE == 1
    // 10K, siftsmall
    #define NR_DPUS 8  // DPU 数量
    #define NUM_CLUSTER 40  // 聚类数量
    #define MAX_RETURNED_PER_DPU 25  // 每个DPU查询返回的最大邻居数量
    #define MAX_QUERY_CLUSTER 15  // sift 需要 9 类，spacev 需要 15 类

#elif DATA_SIZE == 2
    // 1M, sift
    #define NR_DPUS 128
    #define NUM_CLUSTER 400
    #define MAX_RETURNED_PER_DPU 11 
    // Recall@100 = 0.94: sift base 查询 19 类，交集查询 17 类，segment 裁剪 17 类
    // Recall@100 = 0.94: spacev base 查询 42 类，交集查询 37 类，segment 裁剪 38 类
    #define MAX_QUERY_CLUSTER 17

#elif DATA_SIZE == 3
    // 10M, sift 10M
    #define NR_DPUS 640
    #define NUM_CLUSTER 1024
    #define MAX_RETURNED_PER_DPU 3 
    #define MAX_QUERY_CLUSTER 21 

#elif DATA_SIZE == 4 
    // 100M, sift 100M
    #define NR_DPUS 2496
    #define NUM_CLUSTER 4096
    #define MAX_RETURNED_PER_DPU 2 
    /**
     * sift   Recall@100 = 
     * 0.84: sift base 13 (0.8465) ，交集 12 + segment (0.8503) ，segment 裁剪 12 (0.8499)
     * 0.86: sift base 15 (0.8679) ，交集 13 + segment (0.8621) ，segment 裁剪 13 (0.8617)
     * 0.88: sift base 17 (0.8850) ，交集 15 + segment (0.8820) ，segment 裁剪 15 (0.8817)
     * 0.90: sift base 20 (0.9045) ，交集 18 + segment (0.9044) ，segment 裁剪 18 (0.9040)
     * 0.92: sift base 24 (0.9236) ，交集 21 + segment (0.9212) ，segment 裁剪 21 (0.9208)
     * 0.94: sift base 29 (0.9403) ，交集 26 + segment (0.9405) ，segment 裁剪 26 (0.9402)
     * 
     * spacev Recall@100 = 
     * 0.84: base 23 (0.8420)，交集 + segment 18 (0.8449)，segment 裁剪 18 (0.8447)
     * 0.86: base 29 (0.8621)，交集 + segment 22 (0.8630)，segment 裁剪 22 (0.8628)
     * 0.88: base 37 (0.8808)，交集 + segment 28 (0.8827)，segment 裁剪 28 (0.8825)
     * 0.90: base 49 (0.9004)，交集 + segment 36 (0.9009)，segment 裁剪 36 (0.9007)
     * 0.92: base 68 (0.9200)，交集 + segment 49 (0.9204)，segment 裁剪 49 (0.9201)
     * 0.94: base 102 (0.9403)，交集 72 + segment (0.9407)，segment 裁剪 72 (0.9404)
     */
    #define MAX_QUERY_CLUSTER 72

#else
    #error "DATA_SIZE must be in [1 - 4]."
#endif


// 定义查询类型
// batch mode: 0 Sequential; 1 Pipeline; 2 Callback; 3 Async Pipeline;
// QUERY_MODE 通过 CMake 传入，host 和 DPU 使用相同的配置。
#if !defined(QUERY_MODE) || QUERY_MODE < 0 || QUERY_MODE > 3
    #error "QUERY_MODE must be one of 0, 1, 2, or 3."
#endif


// PushPackageHeader 头部（用于 DPU 解析），8 bytes
typedef struct {
    uint8_t option_type; // 操作类型：1=原始数据存储，2=原始数据查询，3=PQ编码存储，4=PQ编码后的查询
    uint8_t is_new; // 对于send_raw_data标志是否是一个新类，1=新类，0=旧类，旧类需要继续后面的位置存储
    uint16_t num_vectors; // 本次传输的向量数量，一次传输肯定不会超过65535
    int data_cluster; // 传输raw data用，记录当前这一批向量所属的 cluster ID
} PushPackageHeader;

typedef struct {
    int vector_id; // 传递raw data时，记录当前向量的ID
    uint16_t cross_cid; // 节点所属的另一个 cluster_id
    uint16_t nonsense;
    uint8_t vector[ALIGN_VECTOR_DIM];  // 传输的向量
} PushPackageBody;  // 未来可以直接在 host 一侧按照 DPU 期望的格式存储并发送，这样 DPU 就不用碎片化读写数据

// 查询开始前，Host-DPU 传输的 DataPackage
typedef struct {
    PushPackageHeader header;  // 头部信息
    PushPackageBody data[]; // 变长数组，存储多个 128 维向量数据和其ID
} DataPackage;


#if (MAX_REPLICA > 2)
    // 每个要查询的向量有多个额外的 id，因此需要一个额外结构体 VectorClusterMap
    // 存储一个向量所属的 cluster id，16 bytes
    typedef struct {
        uint16_t ccount;  // 每一个点所属的cluster数量，如有需要可改成uint8_t
        uint16_t cids[MAX_REPLICA]; // 每个点所属的cluster id列表
    } __attribute__((aligned(8))) VectorClusterMap;

    // 传输raw data的结构体，4+4+16+128/104=152/128 bytes
    typedef struct {
        int vector_id; // 传递raw data时，记录当前向量的ID，未来可移动到 vector_cids 中，减少8字节开销
        uint32_t nonsense; 
        VectorClusterMap vector_cids;
        uint8_t vector[ALIGN_VECTOR_DIM];  // 传输的向量
    } __attribute__((aligned(8))) QueryPackageBody;
    
#else 
    // 传输query的结构体，4+2+2*MAX_QUERY_CLUSTER+10*MAX_QUERY_CLUSTER+4+100=480 bytes (MAX_QUERY_CLUSTER上限按37计算)
    typedef struct {
        int query_id; // 传递raw data时，记录当前向量的ID
        uint8_t split_index;  // 交叉查询的cluster的起始index
        uint8_t qc_num; // 实际待查询的cluster数量
        uint16_t cluster_ids[MAX_QUERY_CLUSTER]; // 待查询的cluster ID
        uint8_t check_dims[MAX_QUERY_CLUSTER][NUM_CHECKPOINTS];  // query vector每个MAX_QUERY_CLUSTER中对检查segment的偏移
        uint8_t vector[ALIGN_VECTOR_DIM] __attribute__((aligned(8))); // 传输的向量
    } __attribute__((aligned(8))) QueryPackageBody;

#endif

// 查询时，Host-DPU 传输的 QueryPackage
typedef struct {
    PushPackageHeader header;  // 头部信息
    QueryPackageBody data[]; // 变长数组，存储多个 128 维向量数据和其ID
} QueryPackage;

// 单个DPU、单个查询向量的结果，4+4+36+36=80 bytes
typedef struct {
    int query_id;    // 该查询的 id
    int return_num;  // 实际返回的数量，预留，暂时无用
    int neighbor_ids[MAX_RETURNED_PER_DPU];  // 最近邻的 ID 
    uint32_t distance[MAX_RETURNED_PER_DPU];  // 最近邻的距离
} __attribute__((aligned(8))) OneQueryResult;

// DPU 查询最后返回给 host 的统计信息，以时钟周期为单位计数
typedef struct {
    uint64_t total_query_num; // 所有线程的查询总数量
    uint64_t calc_time;  // DPU计算距离的时间 
    uint64_t heap_time;  // 堆操作的时间
    uint64_t cross_time;  // WRAM读写时间
    uint64_t wait_time;  // 最快线程的等待时间
    uint64_t total_time; // DPU查询总运行时间
} StatsInfo;



#endif // GLOBAL_HH
