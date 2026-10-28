#ifndef MAIN_HH
#define MAIN_HH

#include "./data_utils/dataset.hh"
#include "./data_utils/kmeans.hh"
#include "./data_utils/cluster_var.hh"
#include "utils.hh"
#include "./host_process/query_batch.hh"
#include "./host_process/query_batch_cpu.hh"

#include "config.hh"


// Ours 查询逻辑（内部加载 cluster 数据）
void my_query_fun(const Config& config, const Dataset& dataset);

// CPU baseline 查询逻辑（内部加载 cluster 数据，不使用 PQ ）
void cpu_query_fun(const Config& config, const Dataset& dataset);


#endif // MAIN_HH
