#ifndef CONFIG_HH
#define CONFIG_HH
#include <string>
#include "utils.hh"
#include "../share/global.h"


/**
 * 使用cxxopts开源库来管理main函数中的参数，为避免该库在.hh中每次都参与编译，利用config.cc进行封装
 */ 


// 枚举定义查询方法
enum class QueryMethod {
    MyQuery, // 自己的查询方法，使用原始数据和聚类结果
    CpuQuery  // 使用 Faiss 库且不使用 PQ 量化的 CPU baseline
};

// 配置结构体：封装所有从命令行获取的配置项
struct Config {
    std::string data_path;
    std::string cluster_path;
    std::string nb_running_dpu_log_path = "./nb_running_dpu.log";
    bool query_debug = false; // 用少量查询来测试正确性
    int query_size = 1; // 用少量查询来测试正确性
    QueryMethod query_method = QueryMethod::MyQuery; 
};

// 利用 cxxopts 库来解析参数，返回 Config 对象
Config parse_args(int argc, char* argv[]);

// 自行封装的参数管理函数
Config parse_args_base(int argc, char* argv[]); 

// 解析 query_method
QueryMethod parse_query_method(const std::string& method_str);


#endif // CONFIG_HH
