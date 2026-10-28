#ifndef QUERY_BATCH_HH
#define QUERY_BATCH_HH
#include <queue>
#include <memory>  // 用于 std::shared_ptr
#include <thread>
#include <cmath>
#include <mutex>
#include <condition_variable>
#include <semaphore>
#include <functional>

#include "host_base.hh"


/**
 * @brief 线程安全队列，用于多线程间构造流水线推送数据
 */
template <typename T>
class ThreadSafeQueue {
public:
    // 构造函数，可以指定队列的最大容量
    ThreadSafeQueue(size_t capacity = 0) : capacity_(capacity), shutdown_(false) {}

    // 向队列中推送一个元素
    void push(T new_value) {
        std::unique_lock<std::mutex> lock(mutex_);
        // 如果队列已满，则等待消费者取走数据
        if (capacity_ > 0) {
            cond_producer_.wait(lock, [this] { return queue_.size() < capacity_ || shutdown_; });
        }
        if (shutdown_) return; // 如果已关闭，则不再添加

        queue_.push(std::move(new_value));
        lock.unlock();
        cond_consumer_.notify_one(); // 通知一个等待的消费者
    }

    // 从队列中弹出一个元素
    // 如果队列为空，则阻塞等待
    // 返回 true 表示成功获取，返回 false 表示队列已关闭且为空
    bool pop(T& value) {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_consumer_.wait(lock, [this] { return !queue_.empty() || shutdown_; });

        if (shutdown_ && queue_.empty()) {
            return false;
        }

        value = std::move(queue_.front());
        queue_.pop();
        lock.unlock();
        cond_producer_.notify_one(); // 通知一个等待的生产者
        return true;
    }
    
    // 关闭队列，唤醒所有等待的线程
    void shutdown() {
        std::unique_lock<std::mutex> lock(mutex_);
        shutdown_ = true;
        lock.unlock();
        cond_producer_.notify_all();
        cond_consumer_.notify_all();
    }

private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cond_producer_;
    std::condition_variable cond_consumer_;
    size_t capacity_; // 0 表示无界队列
    bool shutdown_;
};


/**
 * @brief 封装查询执行所需的所有上下文信息
 */
struct QueryContext {
    // --- 拥有的状态 (Owned State) ---
    // 这些对象在 QueryContext 被创建时一同被创建
    DpuStats dpu_stats;
    HostTimingStats host_stats;
    FlatMatrix<int> topk_results;
    double host_prep_time_us = 0.0;
    const int qc_num;
    
    // --- 引用的外部数据 (Referenced State) ---
    // 这些数据由外部传入，生命周期比 QueryContext 更长
    dpu::DpuSet& system;
    const FlatMatrix<DATA_TYPE>& query_vectors;
    const FlatMatrix<float>& cluster_var_list;
    const FlatMatrix<float>& centroids;
    const FlatMatrix<int32_t>& ground_vids;

    // --- 计时信息 ---
    const std::chrono::steady_clock::time_point start_time;

    // 构造函数，用于方便地创建 Context 对象
    QueryContext(
        dpu::DpuSet& sys,
        const FlatMatrix<DATA_TYPE>& qv,
        const FlatMatrix<float>& c_list,
        const FlatMatrix<float>& cen,
        const FlatMatrix<int32_t>& ground,
        const int qc_num,
        const std::chrono::steady_clock::time_point& start
    ) : 
        dpu_stats(get_dpu_clock(sys)),
        topk_results(qv.size(), TOP_K),
        qc_num(qc_num),
        system(sys),
        query_vectors(qv),
        cluster_var_list(c_list),
        centroids(cen),
        ground_vids(ground),
        start_time(start)
    {}
};


/**
 * @brief 定义查询执行引擎的函数签名，接收一个指向 QueryContext 的 shared_ptr。
 */
using ExecutionEngine = std::function<void(std::shared_ptr<QueryContext> context)>;


/**
 * @brief 通用查询执行器
 * 负责所有查询共有的 初始化 和 收尾统计 逻辑。
 */
class QueryExecutor {
public:
    void run(
        dpu::DpuSet& system,
        const FlatMatrix<DATA_TYPE>& query_vectors,
        const FlatMatrix<float>& cluster_var_list,
        const FlatMatrix<float>& centroids,
        const FlatMatrix<int32_t>& ground_truth_vids,
        const std::string& dpu_log_filename,
        const int qc_num,
        const ExecutionEngine& engine
    ) 
    {
        auto global_start_time = std::chrono::steady_clock::now();

        // --- 1. 创建共享的上下文对象 ---
        auto context = std::make_shared<QueryContext>(
            system, query_vectors, cluster_var_list, centroids, ground_truth_vids, qc_num, global_start_time
        );
        
        // --- 2. 调用注入的执行引擎，传递共享指针 ---
        engine(context);

        context->host_stats.global_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - context->start_time
        );

        // --- 3. 输出 DPU 和 host 的时间统计结果，输出 active dpu number 日志 ---
        const int total_query_num = query_vectors.size();
        const int total_batches = (total_query_num + BATCH_SIZE - 1) / BATCH_SIZE;

        context->host_stats.print<std::chrono::milliseconds>(total_query_num, total_batches);
        context->dpu_stats.print<std::chrono::milliseconds>(total_batches);
        calculate_accuracy(ground_truth_vids, context->topk_results, 0, total_query_num, true);

        std::vector<std::vector<uint64_t>> dpu_time_buffers(NR_DPUS, std::vector<uint64_t>(MAX_DPU_TIME_STAMP, 0));
        pull_batch_time_stamp(system, dpu_time_buffers);
        generate_dpu_activity_log(dpu_log_filename, dpu_time_buffers, total_batches, context->dpu_stats.clocks_per_sec, 
                                  context->host_prep_time_us, context->host_stats.global_time.count());
    }
};


/**
 * @brief 执行查询的函数，根据 query_mode 来选择对应的查询方法并执行。
 * @param system DPU set 对象
 * @param query_vectors 待发送的查询向量
 * @param cluster_var_list 每个聚类下的方差列表
 * @param centroids 每个聚类的质心
 * @param ground_truth_vids 真实 topk 向量的 id
 * @param dpu_log_filename 保存 active dpu number 日志的路径
 * @param query_mode 查询模式
 * @param qc_num 当前这轮查询的 nprobe，不超过 MAX_QUERY_CLUSTER
 */
void execute_query(dpu::DpuSet& system,
                   const FlatMatrix<DATA_TYPE>& query_vectors, 
                   const FlatMatrix<float>& cluster_var_list, 
                   const FlatMatrix<float>& centroids, 
                   const FlatMatrix<int32_t>& ground_truth_vids, 
                   const std::string &dpu_log_filename, 
                   int query_mode,
                   int qc_num = MAX_QUERY_CLUSTER);


/**
 * @brief 顺序查询函数。
 * host 准备查询元数据，向 DPU 发送查询向量和元数据，收回查询结果，统计查询准确率。
 */
void run_synchronous_query(std::shared_ptr<QueryContext> ctx);


/**
 * @brief 同步的流水线查询函数。
 * DPU 计算的过程中，host 准备下一轮查询的元数据、统计上一轮的准确率。
 */
void run_sync_pipeline_query(std::shared_ptr<QueryContext> ctx);


/**
 * @brief 异步查询函数。
 * host 准备元数据发送给 DPU 查询，DPU 算完后调用 callback 通知 host 收回结果并统计准确率。
 */
void run_async_query(std::shared_ptr<QueryContext> ctx);


/**
 * @brief 异步+流水线查询函数。
 * host 与 dpu 的 IO 交互放在一起，dpu 查询和 host 准备下批数据、rerank 上批数据组成流水线。
 */
void run_async_pipeline_query(std::shared_ptr<QueryContext> ctx);


/**
 * @brief 异步+流水线查询函数。
 * 此版本的 IO 线程任务更纯粹，dpu 查询和 host 准备下批数据、rerank 上批数据组成流水线。
 * 适用于 DPU 一侧计算量大的情况，DPU 时间显著变长，但 host 时间相对恒定，因此把更多的任务交给 host 线程。
 */
void run_async_pipeline_query2(std::shared_ptr<QueryContext> ctx);


#endif // QUERY_BATCH_HH
