#include "./utils.hh"
#include <sstream>
#include <iostream>


const std::vector<uint8_t>& get_qc_num_sweep(bool cpu_baseline)
{
    #if (DATA_SIZE == 4 && TOP_K == 10 && DATA_NAME == 0)
        static const std::vector<uint8_t> values = {8, 9, 11, 13, 15, 19};
    #elif (DATA_SIZE == 4 && TOP_K == 10 && DATA_NAME == 1)
        static const std::vector<uint8_t> values = {1, 2, 3};
    #elif (DATA_SIZE == 4 && TOP_K == 100 && DATA_NAME == 0)
        static const std::vector<uint8_t> cpu_values = {11, 13, 15, 18, 21, 27};
        #if (IS_BASELINE == true)
            static const std::vector<uint8_t> values = {13, 15, 17, 20, 24, 29};
        #else
            static const std::vector<uint8_t> values = {12, 13, 15, 18, 21, 26};
        #endif
        if (cpu_baseline) return cpu_values;
    #elif (DATA_SIZE == 4 && TOP_K == 100 && DATA_NAME == 1)
        static const std::vector<uint8_t> cpu_values = {21, 27, 35, 47, 65, 97};
        #if (IS_BASELINE == true)
            static const std::vector<uint8_t> values = {23, 29, 37, 49, 68, 102};
        #else
            static const std::vector<uint8_t> values = {18, 22, 28, 36, 49, 72};
        #endif
        if (cpu_baseline) return cpu_values;
    #else
        static const std::vector<uint8_t> values = {MAX_QUERY_CLUSTER};
    #endif
    
    return values;
}


float weighted_dist_to_cluster(const std::vector<float>& dists, int valid_len)
{
    float weight = 1.0f;
    float sum_dist = 0.0f;
    float sum_weight = 0.0f;
    float factor;
    if (DATA_NAME == 0) factor = 1.5f;
    else factor = 0.8f;

    // 从倒数第二个向前遍历，前后元素的加权比例改为二者距离的比值，可选裁切
    for (int i = valid_len - 2; i >= 0; --i) {
        float ratio = dists[i + 1] / dists[i];
        ratio = 1 + (ratio - 1) * factor;

        // 为了避免权重变化过大，并且前后差异过小，采取 clamp 
        ratio = std::clamp(ratio, 1.0f, 3.0f);  // 允许最多3x增幅

        weight *= ratio;
        sum_dist += dists[i] * weight;
        sum_weight += weight;
    }
    return sum_dist / sum_weight;
}


std::string build_cluster_filename_base(const std::string& cluster_path, int num_cluster) 
{
    const int data_size = get_dataset_size();
    const std::string dataset_name = get_dataset_name();
    
    std::stringstream ss;
    ss << cluster_path << dataset_name;

    // 决定文件名中与大小 (size) 相关的部分
    std::string size_unit;
    switch (DATA_NAME) {
        case 0:
            // sift 10M - 100M
            if (DATA_SIZE == 3 || DATA_SIZE == 4) size_unit = "M";
            break;
        case 1:
            // spacev 10K
            if (DATA_SIZE == 1) size_unit = "K";
            // spacev 1M - 100M
            else size_unit = "M";
            break;
    }

    if (!size_unit.empty()) {
        // bigann / spacev
        ss << data_size << size_unit << "-";
    } else {
        // siftsmall / sift
        ss << "-";
    }

    ss << num_cluster;
    return ss.str();
}


