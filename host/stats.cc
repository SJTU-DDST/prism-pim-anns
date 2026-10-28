#include "stats.hh"


void generate_dpu_activity_log(const std::string& dpu_log_filename, 
                               const std::vector<std::vector<uint64_t>>& dpu_time_buffers, 
                               const int total_batches, 
                               const uint32_t clocks_per_sec, 
                               const uint64_t host_prep_time_us, 
                               const uint64_t total_host_duration_us)
{
    std::ofstream outfile_active_num(dpu_log_filename);
    if (!outfile_active_num.is_open()) {
        std::cerr << "Error: Could not open file " << dpu_log_filename << " for writing." << std::endl;
        return;
    }

    // 预计算每个采样区间对应的 DPU 周期数
    // 使用 double 确保精度，防止 clocks_per_sec * SAMPLE_INTERVAL_US 溢出
    const double cycles_per_interval_double = (double)clocks_per_sec / 1000000 * (double)SAMPLE_INTERVAL_US;
    
    // 必须保证每个区间至少有 1 个 cycle，否则会导致除零
    if (cycles_per_interval_double < 1.0) {
         std::cerr << "Error: SAMPLE_INTERVAL_US is too small. "
                   << "Resulting cycles_per_interval is less than 1." << std::endl;
         outfile_active_num.close(); // 确保关闭文件
         return;
    }
    const uint64_t cycles_per_interval = static_cast<uint64_t>(cycles_per_interval_double);


    int abnormal_dpu_num = 0;  // 记录存在异常数据的 DPU 数量
    bool is_dpu_abnormal = false;

    // 1. 根据Host总时长减去最开始DPU未运行的区段，创建时间区间计数数组
    size_t num_intervals = static_cast<size_t>((total_host_duration_us - host_prep_time_us) / SAMPLE_INTERVAL_US);
    std::vector<uint32_t> active_counts(num_intervals, 0);

    // 2. 遍历所有DPU的时间记录，填充计数数组
    for (int dpu_id = 0; dpu_id < NR_DPUS; ++dpu_id) {
        is_dpu_abnormal = false;
        // 每个 batch 有一对 start/end 时间戳
        for (size_t i = 0; i < (2 * total_batches); i += 2) {
            uint64_t start_cycle = dpu_time_buffers[dpu_id][i];
            uint64_t end_cycle = dpu_time_buffers[dpu_id][i + 1];

            // 对返回异常数据的 DPU 的处理
            if (end_cycle <= start_cycle) {
                // std::cout << "DPU_ID: " << dpu_id << " has error time stamp, start_cycle: " 
                //           << start_cycle << " , end_cycle: " << end_cycle << std::endl;
                if (is_dpu_abnormal == false) {
                    is_dpu_abnormal = true;
                }
                continue; 
            }

            // 计算区间索引，即 [start, end) 区间覆盖的索引范围
            size_t start_interval = start_cycle / cycles_per_interval;
            size_t end_interval = (end_cycle - 1) / cycles_per_interval + 1;

            // 在有效区间内增加计数
            for (size_t j = start_interval; j < end_interval && j < num_intervals; ++j) {
                active_counts[j]++;
            }
        }
        if (is_dpu_abnormal == true) {
            abnormal_dpu_num++;
        }
    }

    // 3. 输出结果到文件，这里补回 host_prep_time_us 的时间，输出前置 0，再输出 dpu 的统计结果
    size_t pre_interval = host_prep_time_us / SAMPLE_INTERVAL_US;
    for (size_t count = 0; count < pre_interval; count++) {
        outfile_active_num << 0 << '\n';
    }
    for (uint32_t count : active_counts) {
        outfile_active_num << count << '\n';
    }

    // 4. 输出 DPU 的实际 active 比例
    uint64_t total_active_num = std::accumulate(active_counts.begin(), active_counts.end(), (uint64_t)0);
    // 注意 intervals 的总数量要包括前置的空闲
    double active_rate = static_cast<double>(total_active_num) / (pre_interval + num_intervals) / NR_DPUS;

    std::cout << "\nGenerated DPU activity log to " << dpu_log_filename << std::endl;
    std::cout << "DPU active rate: " << active_rate << std::endl;
    std::cout << "Abnormal dpu num: " << abnormal_dpu_num << std::endl;

    // 注意：对于小数据集 (1M及以下) 务必要将 SAMPLE_INTERVAL_US 调小 
    // 否则可能因为一个区间内 DPU 做了多个批次，导致最终计算出的 DPU active rate > 1 
    // 同时注意，如果 SAMPLE_INTERVAL_US 过小，也可能导致 DPU 一侧计时的 MAX_DPU_TIME_STAMP 溢出
    if (active_rate >= 1.0) {
        std::cerr << "\n[WARNING] DPU active rate > 1.0! This indicates a measurement or statistics error." << std::endl;
        std::cerr << "Possible causes to check:" << std::endl;
        std::cerr << "1. Buffer Overflow (Most Likely): Check if 'dpu_time_stats[MAX_DPU_TIME_STAMP]' overflowed on the DPU side." << std::endl;
        std::cerr << "   (Total batches: " << total_batches << " vs Buffer capacity)" << std::endl;
        std::cerr << "2. Sampling Granularity: For small datasets (<= 1M), ensure 'SAMPLE_INTERVAL_US' is small enough (currently " 
                  << SAMPLE_INTERVAL_US << "us)." << std::endl;
    }
}

