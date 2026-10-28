#ifndef DATASET_HH
#define DATASET_HH

#include <vector>
#include <cstdint>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <fstream>
#include "../config.hh"
#include "../utils.hh"
#include "flat_matrix.hh"


struct Dataset {
    FlatMatrix<DATA_TYPE> base_vectors;
    FlatMatrix<DATA_TYPE> query_vectors;
    FlatMatrix<int32_t> ground_truth_vids;  // ground_truth的ID部分
    FlatMatrix<float> ground_truth_dists;  // ground_truth的距离部分
    std::string dataset_name;
};

// 加载数据集
Dataset load_dataset(const Config& config);

// 读取 .fvecs 文件 (float)
FlatMatrix<float> read_fvecs(const std::string& filename);

// 读取 .ivecs 文件 (int)
FlatMatrix<int> read_ivecs(const std::string& filename);

// 读取 .bvecs 文件 (unsigned char)，参数data_size控制读取量，默认-1表示全部读入
FlatMatrix<uint8_t> read_bvecs(const std::string& filename, int read_size);
                                             
// 读取 .bvecs 文件 (unsigned char)，全部读入
FlatMatrix<uint8_t> read_bvecs(const std::string& filename); 

// 读取 .bin 文件 (signed char)，spacev 数据集的 data
FlatMatrix<int8_t> read_raw_data_bin(const std::string& filename);

// 读取 .bin 文件 (signed char)，spacev 数据集的 ground truth
std::pair<FlatMatrix<int32_t>, FlatMatrix<float>> 
read_ground_truth_bin(const std::string& filename);


/**
 * @brief 将一个 FlatMatrix<float> 高效且安全地转换为 FlatMatrix<uint8_t>。
 * 该函数采用并行化处理，并对数值进行四舍五入和范围限制。
 * * @param source 输入的 float 类型矩阵。
 * @param destination 输出的 uint8_t 类型矩阵（其内容将被覆盖）。
 */
void convert_vectors_to_uint8(const FlatMatrix<float>& source, 
                              FlatMatrix<uint8_t>& destination);

#endif // DATASET_HH