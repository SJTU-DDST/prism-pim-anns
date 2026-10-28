#ifndef K_MEANS_HH
#define K_MEANS_HH

#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <random>
#include <unordered_set>
#include <thread>
#include "../../share/global.h"
#include "../utils.hh"
#include "flat_matrix.hh"


// KMeans 结果结构体
struct KMeansResult {
    int K = 0;
    FlatMatrix<float> centroids;
    JaggedArray<int> clusters;  // key: cluster_id, value: list of point_ids
    JaggedArray<int> multi_labels;  // key: point_id, value: list of cluster_ids
};

/**
 * @brief 读入指定数据集的聚类结果文件，如果没有成功读入，则重新进行聚类。
 *        数据集的参数目前需要在本函数内部进行调整。
 * @param vectors 读入的 float 类型向量数据
 * @param cluster_path 预先聚类保存的路径
 * @param kmeans_iterations （可选参数）K-means迭代次数（例如 25）
 * @param epsilon （可选参数）决定是否对边界节点进行复制的距离参数
 * @return 存有Kmeans结果的结构体
 */
KMeansResult 
load_clustering_result(const FlatMatrix<DATA_TYPE>& vectors, 
                       const std::string& cluster_path, 
                       int kmeans_iterations, float epsilon);

/**
 * @brief 根据参数，获取 cluster_filename 
 * @return 如果文件存在返回 true，不存在返回 false
 */
bool get_cluster_filename(std::string& cluster_filename, const std::string& cluster_path, 
                          int num_cluster, float epsilon, int min_dup, int max_dup); 

/**
 * @brief 采用 faiss 库的多标签聚类方法，可以叠加边界节点复制策略。
 * @param data 输入待聚类的向量数据
 * @param labels 数据聚类的标签结果，可以是多标签的
 * @param centroids 聚类的质心结果
 * @param iterations 迭代次数
 * @param epsilon 控制节点复制的距离阈值，对边界节点采取复制策略（可选参数）
 */
void faiss_kmeans_with_duplication(const FlatMatrix<DATA_TYPE>& data,
                                   JaggedArray<int>& labels,
                                   FlatMatrix<float>& centroids,
                                   int iterations, float epsilon);

/**
 * @brief 将多聚类标签结果写入文件，根据 labels 转换出 clusters 下的节点 id。
 *        保存形式：第一行保存类别数量K，后面K行保存K个质心；
 *                  后面K行中每行写一个类下的所有 vector id，共有 2K+1 行。
 * @param filename 写入的文件名称
 * @param result 存有Kmeans结果的结构体
 */
void save_kmeans_results(const std::string& filename, const KMeansResult& result);

/**
 * @brief 从文件中读取多聚类标签结果。
 * @param filename 写入的文件名称
 * @return 存有Kmeans结果的结构体
 */
KMeansResult load_kmeans_results(const std::string& filename);

/**
 * @brief 由 multi_labels 构建 cluster。
 * @param clusters 待构建的 cluster
 * @param multi_labels 输入的多标签结果
 * @param K 类别数量
 */
void create_clusters_from_multi_labels(JaggedArray<int>& clusters, const JaggedArray<int>& multi_labels, int K);

#endif // K_MEANS_HH
