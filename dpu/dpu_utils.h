#ifndef DPU_UTILS_H
#define DPU_UTILS_H

#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <defs.h>
#include "../share/global.h"
#include "dpu_global.h"


static inline int16_t bitwise_abs(DATA_TYPE a, DATA_TYPE b) {
    int16_t diff = (int16_t)a - (int16_t)b;
    return (diff ^ (diff >> 15)) - (diff >> 15);
}

static inline uint8_t absolute_diff(DATA_TYPE a, DATA_TYPE b) {
    return (a ^ b) - (((a ^ b) & (a - b)) >> 7);
}

static const uint16_t SQUARE_LUT[256] = {
    0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 
    256, 289, 324, 361, 400, 441, 484, 529, 576, 625, 676, 729, 784, 841, 900, 961, 
    1024, 1089, 1156, 1225, 1296, 1369, 1444, 1521, 1600, 1681, 1764, 1849, 1936, 2025, 2116, 2209, 
    2304, 2401, 2500, 2601, 2704, 2809, 2916, 3025, 3136, 3249, 3364, 3481, 3600, 3721, 3844, 3969, 
    4096, 4225, 4356, 4489, 4624, 4761, 4900, 5041, 5184, 5329, 5476, 5625, 5776, 5929, 6084, 6241, 
    6400, 6561, 6724, 6889, 7056, 7225, 7396, 7569, 7744, 7921, 8100, 8281, 8464, 8649, 8836, 9025, 
    9216, 9409, 9604, 9801, 10000, 10201, 10404, 10609, 10816, 11025, 11236, 11449, 11664, 11881, 12100, 12321, 
    12544, 12769, 12996, 13225, 13456, 13689, 13924, 14161, 14400, 14641, 14884, 15129, 15376, 15625, 15876, 16129, 
    16384, 16641, 16900, 17161, 17424, 17689, 17956, 18225, 18496, 18769, 19044, 19321, 19600, 19881, 20164, 20449, 
    20736, 21025, 21316, 21609, 21904, 22201, 22500, 22801, 23104, 23409, 23716, 24025, 24336, 24649, 24964, 25281, 
    25600, 25921, 26244, 26569, 26896, 27225, 27556, 27889, 28224, 28561, 28900, 29241, 29584, 29929, 30276, 30625, 
    30976, 31329, 31684, 32041, 32400, 32761, 33124, 33489, 33856, 34225, 34596, 34969, 35344, 35721, 36100, 36481, 
    36864, 37249, 37636, 38025, 38416, 38809, 39204, 39601, 40000, 40401, 40804, 41209, 41616, 42025, 42436, 42849, 
    43264, 43681, 44100, 44521, 44944, 45369, 45796, 46225, 46656, 47089, 47524, 47961, 48400, 48841, 49284, 49729, 
    50176, 50625, 51076, 51529, 51984, 52441, 52900, 53361, 53824, 54289, 54756, 55225, 55696, 56169, 56644, 57121, 
    57600, 58081, 58564, 59049, 59536, 60025, 60516, 61009, 61504, 62001, 62500, 63001, 63504, 64009, 64516, 65025
};



// // F: 定点数的小数位数。8位精度足够。
// static const int F = 8;

// // 高性能定点近似log2函数 (无乘除)
// // 返回 log2(n) * 2^F
// static inline uint32_t approx_log2_fixed_point(uint32_t n) {
//     if (n == 0) return 0;

//     // p = floor(log2(n))
//     uint32_t p = 31 - __builtin_clz(n);

//     // 提取小数部分，执行 (x - 2^p) << F >> p
//     uint32_t x_minus_2p = n - (1U << p);
//     uint32_t fraction = (x_minus_2p << F) >> p;

//     return (p << F) + fraction;
// }

// // 统一的、可扩展的阈值计算函数
// // b: 当前段编号 (0 to N-1)
// // N: NUM_CHECKPOINTS
// // max_allowed: 堆顶最大距离
// static inline uint32_t calculate_unified_threshold(int b, int N, uint32_t max_allowed) {
//     // 参数A，用于调整曲线形状
//     const int A = 2;

//     // 计算分子和分母
//     // 增加一个小的线性项 b/16 来抬高曲线，满足更严格的约束
//     uint32_t numerator_log = approx_log2_fixed_point(b + A);
//     uint32_t numerator = numerator_log + (b << (F - 4)); // b/16 * 2^F

//     uint32_t denominator_log = approx_log2_fixed_point((N - 1) + A);
//     uint32_t denominator = denominator_log + ((N - 1) << (F - 4));

//     if (denominator == 0) return max_allowed;

//     // 使用64位中间变量防止溢出
//     uint64_t temp = (uint64_t)max_allowed * numerator;
//     uint32_t result = temp / denominator;
    
//     // 最终阈值不能超过max_allowed
//     return (result > max_allowed) ? max_allowed : result;
// }


// 对应每个 segment 的缩放因子（整数乘法 + 右移）
#if NUM_CHECKPOINTS == 8
    // 8 个阈值分别是 24/64, 34/64, 42/64, 48/64, 54/64, 60/64, 63/64, 1 
    static const uint8_t THRESH_NUM[NUM_CHECKPOINTS]   = {24, 34, 42, 48, 54, 60, 63, 64};

#elif NUM_CHECKPOINTS == 10
    // 10 个阈值分别是 16/64, 24/64, 30/64, 36/64, 42/64, 48/64, 52/64, 58/64, 63/64, 1 
    static const uint8_t THRESH_NUM[NUM_CHECKPOINTS]   = {16, 24, 30, 36, 42, 48, 52, 58, 63, 64};
#endif


// 使用查找表计算欧几里得距离的平方
static inline uint32_t l2_distance_lut(DATA_TYPE *vec1, DATA_TYPE *vec2) 
{
    uint32_t sum = 0;
    for (int i = 0; i < VECTOR_DIM; ++i) {
        // sum += SQUARE_LUT[bitwise_abs(vec1[i], vec2[i])]; // 查表获取平方值，2149 6724
        sum += SQUARE_LUT[abs((int16_t)vec1[i] - (int16_t)vec2[i])]; // 用abs函数和自己写的bitwise差不多快，2149 7100
    }
    return sum; 
}

static inline uint32_t l2_distance_lut_early_exit(DATA_TYPE *vec1, DATA_TYPE *vec2)
{
    uint32_t sum = 0;
    const uint32_t max_allowed = thread_max_dist[me()];
    for (int i = 0; i < VECTOR_DIM; ++i) {
        sum += SQUARE_LUT[abs(vec1[i] - vec2[i])];
        if (sum >= max_allowed)  // 大于堆顶的最大距离，提前退出，减少计算量
            return UINT32_MAX;
    }
    return sum;
}

static inline uint32_t l2_distance_max_check(DATA_TYPE *vec1, DATA_TYPE *vec2, uint8_t priority_dim)
{
    uint32_t sum = 0;
    const uint32_t max_allowed = thread_max_dist[me()];
    // === 先从最大的维度开始往后算起 ===
    for (int i = priority_dim; i < VECTOR_DIM; ++i) {
        sum += SQUARE_LUT[abs(vec1[i] - vec2[i])];
        if (sum >= max_allowed)
            return UINT32_MAX;
    }
    // === 再从头算到最大的维度 ===
    for (int i = 0; i < priority_dim; ++i) {
        sum += SQUARE_LUT[abs(vec1[i] - vec2[i])];
        if (sum >= max_allowed)
            return UINT32_MAX;
    }
    return sum;
}

static inline uint32_t l2_distance_segment_check(DATA_TYPE *vec1, DATA_TYPE *vec2, uint8_t *dims_segment)
{
    uint32_t sum = 0;
    const uint32_t max_allowed = thread_max_dist[me()];
    // 先从方差大的段开始计算
    for (int b = 0; b < NUM_CHECKPOINTS; ++b) {
        int start = dims_segment[b];  // 计算每一个段内的距离
        int end = start + BUCKET_SIZE;
        for (int d = start; d < end; ++d) {
            sum += SQUARE_LUT[abs(vec1[d] - vec2[d])];
        }
        if (sum >= max_allowed)  // 超过阈值，就提前退出
            return UINT32_MAX;
    }
    return sum;
}

// static inline uint32_t l2_distance_segment_compact(DATA_TYPE *vec1, DATA_TYPE *vec2, uint8_t *dims_segment)
// {
//     uint32_t sum = 0;
//     const uint32_t max_allowed = thread_max_dist[me()];

//     // 先从方差大的段开始计算
//     for (int b = 0; b < NUM_CHECKPOINTS; ++b) {
//         // 硬编码约束，根据 sift 维持 Recall = 0.95 时的统计结果得到
//         int start = dims_segment[b];  // 计算每一个段内的距离
//         int end = start + BUCKET_SIZE;
//         for (int d = start; d < end; d += 2) {
//             uint8_t d1 = abs(vec1[d] - vec2[d]);
//             uint8_t d2 = abs(vec1[d+1] - vec2[d+1]);
//             sum += COMBINED_LUT[(d1 << 6) + d2];

//             // 与当前 segment 的阈值进行比较，超过则退出
//             if (sum > max_allowed) {
//                 return UINT32_MAX;
//             }
//         }
//     }
//     return sum;
// }


// 线性阈值过滤，先算小方差的部分，配合线性阈值去筛选。理论上效果应该还可以，实际效果比较差
// 因为我们这里计算的方差大小是一个数据集中的、统计意义上的，不能保证所有数据都适用
// 因此会损害大约20%的数据的查询准确率，只有大约74%
static inline uint32_t l2_distance_segment_linear_check(DATA_TYPE *vec1, DATA_TYPE *vec2, uint8_t *dims_segment) 
{
    uint32_t sum = 0; 
    const uint32_t base_max_allowed = thread_max_dist[me()] >> 3;
    for (int b = NUM_CHECKPOINTS - 1; b > 0; --b) {
        uint32_t curr_threshold = base_max_allowed * (NUM_CHECKPOINTS - b);
        int start = dims_segment[b];
        int end = start + BUCKET_SIZE;
        for (int d = start; d < end; ++d) {
            sum += SQUARE_LUT[abs(vec1[d] - vec2[d])];
            if (sum > curr_threshold) {  
                return UINT32_MAX;
            }
        }
    }
    return sum;
}


#endif // DPU_UTILS_H