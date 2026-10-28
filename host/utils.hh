#ifndef UTILS_HH
#define UTILS_HH
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cassert>
#include <string>
#include <immintrin.h> // 使用AVX/AVX2指令
#include "../share/global.h"


// 流水线各阶段的缓冲深度
constexpr int PIPELINE_DEPTH = 1; 


// 编译时计算对齐到指定边界的函数
constexpr int align_to(int value, int alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

// 使用 inline 关键字，使得每个源文件 include 时定义链接到同一个
// 默认查询 batch_size
inline constexpr int BATCH_SIZE = 128;
static_assert(
    BATCH_SIZE >= 32,
    "ERROR: need larger BATCH_SIZE, else dpu side dpu_time_stats[MAX_DPU_TIME_STAMP] may overflow!"
);  // 例如 10000 个查询，定义 MAX_DPU_TIME_STAMP 为 1024，只能存储512个批次的时间，DPU用了16个线程，因此至少开32

// 编译时计算对齐到 128 B 
inline constexpr int QUERY_PUSH_SIZE = align_to(
    sizeof(PushPackageHeader) + sizeof(QueryPackageBody) * BATCH_SIZE, 128
);
static_assert(
    QUERY_PUSH_SIZE < MAX_PUSH_BUF_SIZE,
    "ERROR: Total query package size exceeds the maximum push buffer size. "
    "Consider reducing BATCH_SIZE."
);

// 默认 raw data 分发 batch_size
inline constexpr int DATA_BATCH_SIZE = 20;
static_assert(
    DATA_BATCH_SIZE % 4 == 0,
    "ERROR: DATA_BATCH_SIZE must be a multiple of 4. "
);
inline constexpr int DATA_PUSH_SIZE = align_to(
    sizeof(PushPackageHeader) + sizeof(PushPackageBody) * DATA_BATCH_SIZE, 32
);
static_assert(
    DATA_PUSH_SIZE < MAX_PUSH_BUF_SIZE,
    "ERROR: Total query package size exceeds the maximum push buffer size. "
    "Consider reducing BATCH_SIZE."
);


inline constexpr int PULL_RESULT_SIZE = align_to(
    sizeof(StatsInfo) + sizeof(OneQueryResult) * BATCH_SIZE, 32
);
static_assert(
    PULL_RESULT_SIZE < MAX_PULL_BUF_SIZE,
    "ERROR: Total query package size exceeds the maximum push buffer size. "
    "Consider reducing BATCH_SIZE."
);  // 修改这里 MAX_PULL_BUF_SIZE 手残写错的bug


// 100M 数据集按 TOP_K 和查询方案选择 nprobe；其他规模保持单次查询。
const std::vector<uint8_t>& get_qc_num_sweep(bool cpu_baseline = false);


/**
 * @brief 计算查询向量到待查询的若干 cluster 质心的加权距离
 * @param dists 查询向量到 cluster 质心的距离平方，须按升序排列（从最近的到最远的）
 * @param valid_len dists 中可访问的有效长度，必须大于 2，实际查询逻辑中一定满足该条件
 * @return 加权平均距离。低于该距离的cluster全查，高于该距离的cluster只查交集
 */
float weighted_dist_to_cluster(const std::vector<float>& dists, int valid_len);


/**
 * @brief 构建有关当前数据集聚类结果的元数据文件名称。
 * @param cluster_path 存储元数据的基本路径
 * @param num_cluster 聚类的数量
 * @return 元数据文件的名称
 */
std::string build_cluster_filename_base(const std::string& cluster_path, int num_cluster);


/**
 * @brief 计算 l2 距离平方值的模板函数，可匹配不同的类型。
 * 返回类型 T3 要求不低于 T1 和 T2 中精度最高的类型，否则可能产生量化误差
 * @param a std::vector 类型向量
 * @param b std::vector 类型向量
 * @return l2 距离的平方值
 */
template <typename T1, typename T2, typename T3>
T3 l2_distance(const std::vector<T1>& a, const std::vector<T2>& b) 
{
    T3 dist = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        T3 diff = static_cast<T3>(a[i]) - static_cast<T3>(b[i]); 
        dist += diff * diff;  // 一般T3精度不低于T1和T2中精度最高的值再转换成 T3
    }
    return dist;
}

/**
 * @brief 计算 l2 距离平方值的模板函数，可匹配不同的类型，接受裸指针和维度的版本。
 * 返回类型 T3 要求不低于 T1 和 T2 中精度最高的类型，否则可能产生量化误差
 * @param a 指向数据 a 的指针
 * @param b 指向数据 b 的指针
 * @param dim 需要计算的维度
 * @return l2 距离的平方值
 */
template <typename T1, typename T2, typename T3>
T3 l2_distance(const T1* a, const T2* b, size_t dim) 
{
    T3 dist = 0;
    for (size_t i = 0; i < dim; ++i) {
        T3 diff = static_cast<T3>(a[i]) - static_cast<T3>(b[i]); 
        dist += diff * diff;
    }
    return dist;
}


inline float l2_distance_squared_avx(const DATA_TYPE* qvec_i8, 
                                     const float* centroid_f32, 
                                     const int dim) 
{
    __m256 sum_vec = _mm256_setzero_ps(); // 累加器，用于存储(float)距离平方和
    int i = 0;

    // --- SIMD 主循环 (一次处理8个维度) ---
    for (; i <= dim - 8; i += 8) {
        // 1. 从内存加载 8 个 signed char (8 * 8 = 64位) 到一个128位的SSE寄存器的低半部分
        //    这里的 reinterpret_cast 是安全的，只是为了让加载指令正确工作
        const __m128i qvec_i8_chunk = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qvec_i8 + i));
        
        // 2. 将这 8 个 数据扩展成 8 个 int32_t / uint32_t，存入一个256位的AVX寄存器
        __m256i qvec_i32_vec;

        // 根据 DATA_NAME 选择正确的转换指令
        #if DATA_NAME == 0 // 0: sift -> uint8_t
            // 使用零扩展 (Zero Extend)
            qvec_i32_vec = _mm256_cvtepu8_epi32(qvec_i8_chunk);
        #else // 1: spacev -> int8_t
            // 使用符号扩展 (Sign Extend)
            qvec_i32_vec = _mm256_cvtepi8_epi32(qvec_i8_chunk);
        #endif
        
        // 3. 将这 8 个 int32_t 转换成 8 个 float
        const __m256 qvec_f32_vec = _mm256_cvtepi32_ps(qvec_i32_vec);
        
        // 4. 从内存加载 8 个质心 float 数据
        const __m256 centroid_f32_vec = _mm256_loadu_ps(centroid_f32 + i);

        // 5. 后续计算与之前完全相同
        const __m256 sub_vec = _mm256_sub_ps(qvec_f32_vec, centroid_f32_vec);
        sum_vec = _mm256_fmadd_ps(sub_vec, sub_vec, sum_vec);
    }

    // 将AVX寄存器中的8个部分和加起来
    float sum_arr[8];
    _mm256_storeu_ps(sum_arr, sum_vec);
    float sum = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3] +
                sum_arr[4] + sum_arr[5] + sum_arr[6] + sum_arr[7];
    
    // --- 标量收尾循环 (处理剩余维度) ---
    for (; i < dim; ++i) {
        // 这里进行逐元素的标量转换和计算
        const float diff = static_cast<float>(qvec_i8[i]) - centroid_f32[i];
        sum += diff * diff;
    }
    
    return sum;
}


// 根据 DATA_NAME 和 DATA_SIZE 宏返回数据集名称
inline const std::string get_dataset_name() 
{
    if (DATA_NAME == 0) {
        switch (DATA_SIZE) {
            case 1: return "siftsmall";
            case 2: return "sift";
            case 3: 
            case 4: return "bigann";
            default: return "siftsmall";
        }
    } else if (DATA_NAME == 1) {
        return "spacev";  // spacev 数据集的 10K, 1M, 100M 都是同一个路径
    }
}

// 根据 DATA_SIZE 宏返回实际的数据大小
inline const int get_dataset_size() 
{
    switch (DATA_SIZE) {
        case 1: return 10;  // 10k
        case 2: return 1;   // 1m
        case 3: return 10;  // 10m
        case 4: return 100; // 100m
        default: return 10;
    }
}

// 根据 QUERY_MODE 宏返回实际的查询方法
inline const std::string get_query_mode() 
{
    switch (QUERY_MODE) {
        case 0: return "Sequential";
        case 1: return "Pipeline";
        case 2: return "Callback";
        case 3: return "Async Pipeline";
        default: return "ERROR query mode!";
    }
}

// Murmur 哈希函数
static inline int hash_function(int key)
{
    key ^= key >> 17;
    key *= 0xed5ad4bb;
    key ^= key >> 11;
    key *= 0xac4c1b51;
    key ^= key >> 15;
    key *= 0x31848bab;
    key ^= key >> 14;
    key &= 0x7FFFFFFF; // 去除最高位的负号
    return key; 
}


#endif // UTILS_HH
