#ifndef STATS_HH
#define STATS_HH
#include <chrono> // 用于统计时间
#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream> // 必须包含这个
#include <numeric>
#include <cmath>
#include "../share/global.h"


static constexpr int SAMPLE_INTERVAL_US = 2000;  // 统计正在运行的dpu数量的采样间隔

using DurationType = std::chrono::microseconds; // host 一侧计时器内部统一使用微秒


// host 计时结构体
struct HostTimingStats {
    DurationType cluster_time{0};
    DurationType segment_time{0};
    DurationType query_time{0};
    DurationType pull_time{0};
    DurationType rerank_time{0};
    DurationType sync_time{0};
    DurationType global_time{0};
    DurationType batch_time{0};  // 所有 batch 处理时间的累计

    // 新增方法：累加batch时间
    void add_batch_time(const DurationType& processing_time) {
        batch_time += processing_time;
    }
    
    // 新增方法：累加BatchDurationType时间
    void add_batch_duration(DurationType duration) {
        batch_time += duration;
    }

    // 模板化的输出函数，单位默认为 milliseconds，可改为 microseconds、seconds 等
    template<typename OutputDuration = std::chrono::milliseconds>
    void print(const int query_num, const int batch_num) const {
        // 将总时间转换为秒计算 QPS
        const double total_seconds = std::chrono::duration<double>(global_time).count();
        const double qps = query_num / total_seconds;

        // 计算平均batch延迟（转换为毫秒显示）
        const double avg_batch_latency = to_output_unit<OutputDuration>(batch_time).count() / batch_num;

        std::cout << "\n***** Query run-time breakdown *****\n";
        std::cout << "Cluster find         : " << to_output_unit<OutputDuration>(cluster_time).count() 
                  << unit_name<OutputDuration>() << "\n";
        std::cout << "Segment (and push)    : " << to_output_unit<OutputDuration>(segment_time).count() 
                  << unit_name<OutputDuration>() << "\n";
        std::cout << "Query                : " << to_output_unit<OutputDuration>(query_time).count() 
                  << unit_name<OutputDuration>() << "\n";
        std::cout << "Pull (and rerank)    : " << to_output_unit<OutputDuration>(pull_time).count() 
                  << unit_name<OutputDuration>() << "\n";
        std::cout << "Rerank time          : " << to_output_unit<OutputDuration>(rerank_time).count() 
                  << unit_name<OutputDuration>() << "\n";
        std::cout << "DPU Sync time        : " << to_output_unit<OutputDuration>(sync_time).count() 
                  << unit_name<OutputDuration>() << "\n";
        std::cout << "Global time          : " << to_output_unit<OutputDuration>(global_time).count() 
                  << unit_name<OutputDuration>() << "\n";
        std::cout << "QPS (queries/second) : " << qps << "\n";
        std::cout << "Avg Batch latency    : " << avg_batch_latency << unit_name<OutputDuration>() << "\n";
    }

private:

    // 将微秒转换为指定的输出单位
    template<typename OutputDuration>
    OutputDuration to_output_unit(const DurationType& us) const {
        return std::chrono::duration_cast<OutputDuration>(us);
    }
    
    // 根据输出类型选择模板名称
    template<typename OutputDuration>
    static const char* unit_name() { 
        if constexpr (std::is_same_v<OutputDuration, std::chrono::nanoseconds>)  return " ns";
        if constexpr (std::is_same_v<OutputDuration, std::chrono::microseconds>) return " us";
        if constexpr (std::is_same_v<OutputDuration, std::chrono::milliseconds>) return " ms";
        if constexpr (std::is_same_v<OutputDuration, std::chrono::seconds>) return " s";
        return " unknown";
    }
};


// host 计时器类，利用 ScopedTimer 的生命周期，记录函数的运行时间
class ScopedTimer {
public:
    using Clock = std::chrono::steady_clock;

    // 构造函数接受微秒引用
    explicit ScopedTimer(DurationType& target)
        : start_(Clock::now()), target_(target) {}

    ~ScopedTimer() {
        auto end = Clock::now();
        // 内部计算使用高精度，然后转换为微秒
        target_ += std::chrono::duration_cast<DurationType>(end - start_);
    }

private:
    Clock::time_point start_;
    DurationType& target_;
};


// host 一侧存储的 DPU 查询最后返回给 host 的统计信息结构体
struct DpuStats {
    uint32_t clocks_per_sec;  // DPU时钟频率，用于将周期数转换为时间
    uint64_t total_query_num = 0;
    double calc_time = 0.0;
    double heap_time = 0.0;
    double cross_time = 0.0;
    double wait_time = 0.0;
    double total_time = 0.0;
    double max_min_time = 0.0;
    std::vector<double> current_batch_times;  // 临时存储当前批次的 total_time 值
    int rejected_dpu_count = 0;  // 记录每个批次中被舍弃的异常统计值的 DPU 数量
    uint64_t total_valid_count = 0; 

    /**
     * @brief DpuStats 构造函数
     * @param clocks_per_sec DPU 的时钟频率
     */
    explicit DpuStats(uint32_t dpu_clocks_per_sec) : clocks_per_sec(dpu_clocks_per_sec) { }

    
    void start_new_batch() {
        current_batch_times.clear();
        current_batch_times.reserve(NR_DPUS);
        rejected_dpu_count = 0;
    }
    

    void finalize_batch() {  
        // 增加一个日志，告知有多少DPU数据是无效的
        // if (rejected_dpu_count > 0) {
        //     std::cout << "Notice: Rejected " << rejected_dpu_count << " DPU stats in this batch due to abnormal values." << std::endl;
        // }

        // 计算当前批次的 DPU 运行时间的平均值
        // double batch_avg = std::accumulate(current_batch_times.begin(), 
        //                                    current_batch_times.end(), 0.0) / current_batch_times.size();
        double max_time = *std::max_element(current_batch_times.begin(), current_batch_times.end());
        double min_time = *std::min_element(current_batch_times.begin(), current_batch_times.end());
        max_min_time += (max_time - min_time);  // 当前批次最慢和最快酷开 DPU 的运行时间的差值
        total_time += max_time; // 以最慢的 DPU 的时间作为当前批次总的 DPU 运行时间
    }


    void add(const StatsInfo* s, int dpu_id) {
        total_query_num += static_cast<uint64_t>(s->total_query_num);  // 累加查询数量

        const uint64_t CYCLE_THRESHOLD = std::numeric_limits<uint32_t>::max();  // 最大阈值选择 UINT32T_MAX
        bool is_abnormal = (s->total_time > CYCLE_THRESHOLD || s->calc_time > CYCLE_THRESHOLD ||
                            s->heap_time > CYCLE_THRESHOLD || s->cross_time > CYCLE_THRESHOLD ||
                            s->wait_time > CYCLE_THRESHOLD);

        if (is_abnormal) {
            rejected_dpu_count++;  // 如果任何一个时间异常，增加拒绝计数器，直接返回
            return;
        }

        // 如果数据正常，则累加所有时间值
        calc_time += static_cast<double>(s->calc_time);
        heap_time += static_cast<double>(s->heap_time);
        cross_time += static_cast<double>(s->cross_time);
        wait_time += static_cast<double>(s->wait_time);
        current_batch_times.push_back(static_cast<double>(s->total_time));
        total_valid_count++;  // 累加有效DPU的总数

        // 注意输出的时间中，batch cycle是DPU总的程序时间，其余时间均为DPU中所有NR_TASKLETS个线程时间的求和
        // std::cout << "DPU " << dpu_id << ", batch cycle:" << s->total_time 
        //           << ", calc_cycle:" << s->calc_time << ", heap_cycle:" << s->heap_time 
        //           << ", cross_cycle:" << s->cross_time << ", wait_cycle:" << s->wait_time << std::endl;
    }


    template <typename DurationType>
    void print(const int batch_num) const {        
        // 将周期按照 DPU 转换为目标单位 (DurationType)
        auto to_unit = [&](double cycles) {
            double seconds = cycles / static_cast<double>(clocks_per_sec);
            std::chrono::duration<double> duration_in_seconds(seconds);
            auto final_duration = std::chrono::duration_cast<DurationType>(duration_in_seconds);
            return final_duration.count();
        };

        // 将周期按照线程转换为目标单位 (DurationType)
        auto to_unit_tasklet = [&](double cycles, const int batch_num) {
            // 1. 先计算出平均每个 tasklet 的时钟周期数
            double avg_cycles = cycles * batch_num / static_cast<double>(total_valid_count) / NR_TASKLETS;
            // 2. 直接调用 to_unit 函数来处理单位转换
            return to_unit(avg_cycles);
        };
        
        std::cout << "\n***** DPU run-time breakdown *****" << std::endl;
        std::cout << "Total visited vector number      : " << total_query_num << std::endl;
        std::cout << "DPU run-time (slowest accum)     : " << to_unit(total_time) << unit_name<DurationType>() << "\n";
        std::cout << "Calculate time (tasklet avg)     : " << to_unit_tasklet(calc_time, batch_num) << unit_name<DurationType>() << "\n";
        std::cout << "Heap time (tasklet avg)          : " << to_unit_tasklet(heap_time, batch_num) << unit_name<DurationType>() << "\n";
        std::cout << "Cluster cross time (tasklet avg) : " << to_unit_tasklet(cross_time, batch_num) << unit_name<DurationType>() << "\n";
        std::cout << "Avg tasklet wait time            : " << to_unit_tasklet(wait_time, batch_num) << unit_name<DurationType>() << "\n";
        std::cout << "DPU imbalance (accumulated)      : " << to_unit(max_min_time) << unit_name<DurationType>() << "\n";
    }

private:
    template <typename DurationType>
    static const char* unit_name() { 
        if constexpr (std::is_same_v<DurationType, std::chrono::nanoseconds>)  return " ns";
        if constexpr (std::is_same_v<DurationType, std::chrono::microseconds>) return " us";
        if constexpr (std::is_same_v<DurationType, std::chrono::milliseconds>) return " ms";
        if constexpr (std::is_same_v<DurationType, std::chrono::seconds>)      return " s";
        return "unknown time type";
    }
};



/**
 * @brief 根据 DPU 原始计时数据生成活跃 DPU 数量的时间序列日志
 * @param dpu_log_filename 输出日志的文件路径
 * @param dpu_time_buffers 从所有 DPU 拉取回来的原始计时数据 (cycle counts)
 * @param total_batches 本次运行的总批次数 (用于确定每个DPU有多少有效时间戳)
 * @param clocks_per_sec DPU 每秒的时钟周期数
 * @param host_prep_time_us Host 从程序开始到启动第一个 DPU 任务所用的准备时间 (us)
 * @param total_host_duration_us Host 测量的从程序开始到结束的总时长 (us), 用于对齐时间轴
 */
void generate_dpu_activity_log(const std::string& dpu_log_filename, 
                               const std::vector<std::vector<uint64_t>>& dpu_time_buffers, 
                               const int total_batches, 
                               const uint32_t clocks_per_sec, 
                               const uint64_t host_prep_time_us, 
                               const uint64_t total_host_duration_us);


#endif  // STATS_HH