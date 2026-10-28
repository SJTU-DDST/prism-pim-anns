#ifndef CLUSTER_VAR_HH
#define CLUSTER_VAR_HH

#include <vector>
#include <cstdint>
#include <algorithm>
#include <cassert>
#include <string>
#include <filesystem>
#include <iostream>
#include <fstream>
#include "../../share/global.h"
#include "../utils.hh"
#include "flat_matrix.hh"


/**
 * @brief 读入当前数据集每个 cluster 内数据在各维度的方差；如果文件不存在，则重新计算。
 * @param cluster_path 文件保存的基本路径
 * @param cluster_ids 每一个 cluster 下的向量 id 列表
 * @param base_vectors 原始数据集
 * @param centroids 聚类得到的质心
 * @return 每一个 cluster 内部数据各维度的方差结果
 */
FlatMatrix<float> 
load_intra_cluster_var(const std::string& cluster_path, 
                       const JaggedArray<int>& cluster_ids, 
                       const FlatMatrix<DATA_TYPE>& base_vectors,
                       const FlatMatrix<float>& centroids);

/**
 * @brief 根文件保存的基本路径和分类数，获取数据集的类内方差文件名称。
 * @param cluster_path 每一个 cluster 下的向量 id 列表
 * @param filename 文件保存的基本路径
 * @param num_cluster 分类的数量
 */
bool get_cluster_var_filename(const std::string& cluster_path, std::string& filename, int num_cluster);

/**
 * @brief 根据文件名称，写入类内方差。
 * @param filename 文件保存的基本路径
 * @param cluster_var_list 类内方差
 */
void save_intra_cluster_var_list(const std::string& filename, const FlatMatrix<float>& cluster_var);

/**
 * @brief 根据文件名称，读入类内方差。
 * @param filename 文件保存的基本路径
 * @return cluster_var_list 类内方差
 */
FlatMatrix<float> load_intra_cluster_var_list(const std::string& filename); 

/**
 * @brief 计算每个 cluster 内数据在各维度的方差。
 * @param cluster_ids 每一个 cluster 下的向量 id 列表
 * @param base_vectors 原始数据集
 * @param centroids 聚类得到的质心
 * @param cluster_var 输出的每一个 cluster 内在数据各维度的方差结果
 */
void compute_intra_cluster_var(const JaggedArray<int>& cluster_ids, 
                               const FlatMatrix<DATA_TYPE>& base_vectors,
                               const FlatMatrix<float>& centroids, 
                               FlatMatrix<float>& cluster_var);


#endif // CLUSTER_VAR_HH