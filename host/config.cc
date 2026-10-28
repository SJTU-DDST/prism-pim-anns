#include "config.hh"
#include "cxxopts.hpp"  // 管理参数
#include <iostream>
#include <stdexcept>


Config parse_args(int argc, char* argv[])
{
    cxxopts::Options options("PIM_ANNS_System", "A PIM-based system for Approximate Nearest Neighbor Search.");

    options.add_options()
        ("b,data_path", "Base path for the dataset",
            cxxopts::value<std::string>()->default_value("/mnt/anns-dataset/"))
        ("c,cluster_path", "Base path for the quantized clusters",
            cxxopts::value<std::string>()->default_value("/mnt/anns-dataset/kmeans-output/"))
        ("d,debug", "Enable query debug mode (use a smaller query set)",
            cxxopts::value<bool>()->default_value("true")->implicit_value("true")) // 保持原始逻辑默认开启
        ("s,query_size", "Number of queries to use in debug mode",
            cxxopts::value<int>()->default_value("10000"))
        ("q,query_method", "Query method: MyQuery or CpuQuery",
            cxxopts::value<std::string>()->default_value("MyQuery"))
        ("l,log_path", "Path to save the number of running DPUs log",
            cxxopts::value<std::string>())
        ("h,help", "Print usage information");

    // 添加对位置参数的支持，使其可以像以前一样 `program ./log.txt` 运行
    options.parse_positional({"log_path"});
    options.positional_help("[log_path]");

    cxxopts::ParseResult result;
    try {
        result = options.parse(argc, argv);
    } catch (const cxxopts::exceptions::exception& e) {
        std::cerr << "Error parsing options: " << e.what() << std::endl;
        std::cout << options.help() << std::endl;
        exit(1);
    }

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    Config config;
    config.data_path = result["data_path"].as<std::string>();
    config.cluster_path = result["cluster_path"].as<std::string>();
    config.query_debug = result["debug"].as<bool>();
    config.query_size = result["query_size"].as<int>();
    
    // 使用辅助函数将字符串转换为枚举
    config.query_method = parse_query_method(result["query_method"].as<std::string>());

    // 处理位置参数或通过 -l/--log_path 传入的参数
    if (result.count("log_path")) {
        config.nb_running_dpu_log_path = result["log_path"].as<std::string>();
    } else {
        // 如果没有该参数，设置一个默认值
        config.nb_running_dpu_log_path = "./nb_running_dpu.log";
    }

    // --- 配置信息打印逻辑 ---
    std::cout << "[Config] Configuration loaded via cxxopts." << std::endl;
    std::cout << " - DPU num log path : " << config.nb_running_dpu_log_path << "\n";
    std::cout << " - NR_DPUS            : " << (NR_DPUS) << "\n";
    std::cout << " - DPU NR_TASKLETS    : " << (NR_TASKLETS) << "\n";
    std::cout << " - dataset name       : " << (get_dataset_name()) << "\n";
    std::cout << " - data size          : " << (get_dataset_size()) << " million (valid when data size >= 1 million)\n";
    std::cout << " - cluster num        : " << (NUM_CLUSTER) << " (may be different in variable data_size)\n";
    std::cout << " - query cluster num  : " << (MAX_QUERY_CLUSTER) << " (may be different)\n";
    std::cout << " - query debug mode   : " << (config.query_debug ? "true" : "false") << "\n";
    std::cout << " - query size         : " << (config.query_size) << " (valid when query debug mode is true)\n";
    std::cout << " - query method       : ";
    switch (config.query_method) {
        case QueryMethod::MyQuery:  std::cout << "MyQuery"; break;
        case QueryMethod::CpuQuery: std::cout << "CpuQuery"; break;
    }
    std::cout << std::endl;
    std::cout << " - query mode         : " << (get_query_mode()) << std::endl;

    return config;
}


QueryMethod parse_query_method(const std::string& method_str) 
{
    std::string lower_str = method_str;
    std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(), ::tolower);

    if (lower_str == "myquery") return QueryMethod::MyQuery;
    if (lower_str == "cpuquery") return QueryMethod::CpuQuery;
    
    // 如果没有匹配项，抛出异常或返回一个默认值并打印警告
    std::cerr << "Warning: Invalid query method '" << method_str << "'. Defaulting to MyQuery." << std::endl;
    return QueryMethod::MyQuery;
}
