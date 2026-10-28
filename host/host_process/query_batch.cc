#include "query_batch.hh"
#include <unistd.h>


void execute_query(dpu::DpuSet& system, 
                   const FlatMatrix<DATA_TYPE>& query_vectors, 
                   const FlatMatrix<float>& cluster_var_list, 
                   const FlatMatrix<float>& centroids, 
                   const FlatMatrix<int32_t>& ground_truth_vids, 
                   const std::string &dpu_log_filename, 
                   int query_mode,
                   int qc_num)
{
    QueryExecutor executor;
    switch (query_mode) {
        case 0:
            executor.run(system, query_vectors, cluster_var_list, centroids, 
                         ground_truth_vids, dpu_log_filename, qc_num, run_synchronous_query);
            break;
        case 1:
            executor.run(system, query_vectors, cluster_var_list, centroids, 
                         ground_truth_vids, dpu_log_filename, qc_num, run_sync_pipeline_query);
            break;
        case 2:
            executor.run(system, query_vectors, cluster_var_list, centroids, 
                         ground_truth_vids, dpu_log_filename, qc_num, run_async_query);
            break;
        case 3:
            executor.run(system, query_vectors, cluster_var_list, centroids, 
                         ground_truth_vids, dpu_log_filename, qc_num, run_async_pipeline_query2);
            break;
        default:
            std::cout << "ERROR QUERY_MODE!" << std::endl;
            return;
    }
}



void run_synchronous_query(std::shared_ptr<QueryContext> ctx)
{
    int query_offset = 0;
    int total_query_num = ctx->query_vectors.size();

    while (query_offset < total_query_num) {
        int remaining_num = total_query_num - query_offset;
        const uint16_t batch_size = std::min(BATCH_SIZE, remaining_num);
        // std::cout << "[Query] query_offset: " << query_offset << ", batch_size " << batch_size 
        //           << ", remaining_num: " << remaining_num << std::endl;

        auto batch = std::make_shared<Batch>();
        batch->allocate(batch_size, query_offset, ctx->query_vectors, ctx->qc_num, ctx->ground_vids);

        // 1. 找到要查询的 cluster
        {
            ScopedTimer timer(ctx->host_stats.cluster_time);
            batch->start_timer();
            find_query_clusters(batch->vectors, ctx->centroids, batch->labels,
                                batch->split_indices, batch->qc_num);
        }

        // 2. 计算每个 segment 的方差
        {
            ScopedTimer timer(ctx->host_stats.segment_time);
            compute_query_segment_scores(batch->vectors, ctx->centroids, batch->labels, ctx->cluster_var_list,
                                        batch->var_score_list, batch->qc_num, 0.5f, 0.5f);
        }

        // 3. 发送给 DPU 查询
        {
            ScopedTimer timer(ctx->host_stats.query_time);
            if (query_offset == 0) {
                auto first_batch_launch = std::chrono::steady_clock::now();
                std::chrono::duration<double, std::micro> offset = first_batch_launch - ctx->start_time;
                ctx->host_prep_time_us = offset.count();
            }
            broadcast_query_batch(ctx->system, batch->vectors, batch->labels, batch->split_indices, 
                                  batch->qc_num, batch->var_score_list, query_offset);
        }

        // 4. 从 DPU 中收回结果并提取 topK，并计算批次准确率
        {
            ScopedTimer timer(ctx->host_stats.pull_time);
            pull_results_and_extract_topk(ctx->system, ctx->topk_results, ctx->dpu_stats, batch->offset, batch->size);
            batch->stop_timer();
            ctx->host_stats.add_batch_time(batch->get_processing_time());
            // calculate_accuracy(ctx->ground_vids, ctx->topk_results, batch->offset, batch->size, false);
        }

        query_offset += batch_size;
    }
}


void run_sync_pipeline_query(std::shared_ptr<QueryContext> ctx)
{
    dpu::DpuSetAsync async_system = ctx->system.async();  
    int query_offset = 0;
    const int total_query_num = ctx->query_vectors.size();
    std::shared_ptr<Batch> batch = nullptr;

    while (query_offset < total_query_num || batch != nullptr) {
        // 4. 拉取上一个 batch 的结果（如果存在）
        if (batch) {
            {
                ScopedTimer timer(ctx->host_stats.pull_time);
                pull_results_and_extract_topk(ctx->system, ctx->topk_results, ctx->dpu_stats, batch->offset, batch->size);
                batch->stop_timer();
                ctx->host_stats.add_batch_time(batch->get_processing_time());
                // calculate_accuracy(ctx->ground_vids, ctx->topk_results, batch->offset, batch->size, false);
            }
        }

        // 如果没有新的查询需要处理，并且已处理完最后一个批次，则退出
        if (query_offset >= total_query_num) break;

        // 构造当前 batch
        int remaining_num = total_query_num - query_offset;
        const uint16_t batch_size = std::min(BATCH_SIZE, remaining_num);
        std::cout << "[Query] query_offset: " << query_offset << ", batch_size " << batch_size 
                  << ", remaining_num: " << remaining_num << std::endl;
        batch = std::make_shared<Batch>();
        batch->allocate(batch_size, query_offset, ctx->query_vectors, ctx->qc_num, ctx->ground_vids);

        // 1. 找到要查询的 cluster
        {
            ScopedTimer timer(ctx->host_stats.cluster_time);
            batch->start_timer();
            find_query_clusters(batch->vectors, ctx->centroids, batch->labels, 
                                batch->split_indices, batch->qc_num);
        }

        // 2. 计算每个 segment 的方差
        {
            ScopedTimer timer(ctx->host_stats.segment_time);
            compute_query_segment_scores(batch->vectors, ctx->centroids, batch->labels, ctx->cluster_var_list,
                                        batch->var_score_list, batch->qc_num, 0.5f, 0.5f);
        }

        async_system.sync();  // 等待上一轮查询完成

        // 3. 发送给 DPU 查询
        {
            ScopedTimer timer(ctx->host_stats.query_time);
            if (query_offset == 0) {
                auto first_batch_launch = std::chrono::steady_clock::now();
                std::chrono::duration<double, std::micro> offset = first_batch_launch - ctx->start_time;
                ctx->host_prep_time_us = offset.count();
            }
            broadcast_query_batch(ctx->system, batch->vectors, batch->labels, batch->split_indices, 
                                  batch->qc_num, batch->var_score_list, query_offset);
        }

        query_offset += batch_size;
    }
}


void run_async_query(std::shared_ptr<QueryContext> ctx)
{
    dpu::DpuSetAsync async_system = ctx->system.async(); 
    int query_offset = 0;
    int total_query_num = ctx->query_vectors.size();
    const int total_batches = (total_query_num + BATCH_SIZE - 1) / BATCH_SIZE;

    // 用于等待所有回调完成的同步原语
    std::atomic<int> remaining_cb = total_batches;
    std::mutex cb_mutex;
    std::condition_variable cb_cv;

    while (query_offset < total_query_num) {
        int remaining_num = total_query_num - query_offset;
        const uint16_t batch_size = std::min(BATCH_SIZE, remaining_num);

        auto batch = std::make_shared<Batch>();
        batch->allocate(batch_size, query_offset, ctx->query_vectors, ctx->qc_num, ctx->ground_vids);

        {
            ScopedTimer timer(ctx->host_stats.cluster_time);
            batch->start_timer();
            find_query_clusters(batch->vectors, ctx->centroids, batch->labels,
                                batch->split_indices, batch->qc_num);
        }

        {
            ScopedTimer timer(ctx->host_stats.segment_time);
            compute_query_segment_scores(batch->vectors, ctx->centroids, batch->labels, ctx->cluster_var_list,
                                        batch->var_score_list, batch->qc_num, 0.5f, 0.5f);
        }

        std::cout << "[Query] query_offset: " << query_offset << ", batch_size " << batch_size 
                  << ", remaining_num: " << remaining_num << std::endl;
        {
            ScopedTimer timer(ctx->host_stats.query_time);
            if (query_offset == 0) {
                auto first_batch_launch = std::chrono::steady_clock::now();
                std::chrono::duration<double, std::micro> offset = first_batch_launch - ctx->start_time;
                ctx->host_prep_time_us = offset.count();
            }
            broadcast_query_batch(ctx->system, batch->vectors, batch->labels, batch->split_indices, 
                                  batch->qc_num, batch->var_score_list, query_offset);
        }

        // 回调函数的第一个参数采用 Lambda 函数
        async_system.call(
            [ctx, batch, &remaining_cb, &cb_mutex, &cb_cv]
            (dpu::DpuSet& system, unsigned /*rank_id*/) {
                
                // 4. 从 DPU 中收回结果并提取 topK，计算准确率可选
                {
                    ScopedTimer timer(ctx->host_stats.pull_time);
                    pull_results_and_extract_topk(system, ctx->topk_results, ctx->dpu_stats, batch->offset, batch->size);
                    batch->stop_timer();
                    ctx->host_stats.add_batch_time(batch->get_processing_time());
                    calculate_accuracy(ctx->ground_vids, ctx->topk_results, batch->offset, batch->size, false);
                }

                // 使用条件变量进行精确的完成通知
                if (--remaining_cb == 0) {
                    std::lock_guard<std::mutex> lock(cb_mutex);
                    cb_cv.notify_one();
                }
            },
            /*IsBlocking=*/false, /*SingleCall=*/true);

        query_offset += batch_size;
    }
    async_system.sync();

    // 高效地等待最后一个回调函数完成
    {
        std::unique_lock<std::mutex> lock(cb_mutex);
        cb_cv.wait(lock, [&]{ return remaining_cb == 0; });
    }
}


void run_async_pipeline_query(std::shared_ptr<QueryContext> ctx)
{
    // --- 1. 初始化 ---
    dpu::DpuSetAsync async_system = ctx->system.async(); 
    const int total_query_num = ctx->query_vectors.size();
    const int total_batches = (total_query_num + BATCH_SIZE - 1) / BATCH_SIZE;
    
    // 用于主线程等待所有任务完成的同步原语
    std::atomic<int> reranked_batches_count = 0;
    std::mutex final_sync_mutex;
    std::condition_variable final_sync_cv;

    std::atomic<int> dpu_callbacks_completed = 0;
    std::mutex callback_mutex;
    std::condition_variable callback_cv;

    // --- 2. 创建连接流水线各阶段的队列 ---
    ThreadSafeQueue<std::shared_ptr<Batch>> queue_prepared(PIPELINE_DEPTH);
    ThreadSafeQueue<std::shared_ptr<Batch>> queue_for_rerank(PIPELINE_DEPTH);

    // --- 线程 1: Producer (生产者) ---
    // 职责: 准备数据, 放入 queue_prepared.
    auto producer_logic = [&]() {
        for (int i = 0; i < total_batches; ++i) {
            int query_offset = i * BATCH_SIZE;
            const uint16_t batch_size = std::min(BATCH_SIZE, total_query_num - query_offset);

            auto batch = std::make_shared<Batch>();
            batch->allocate(batch_size, query_offset, ctx->query_vectors, ctx->qc_num, ctx->ground_vids);

            {
                ScopedTimer timer(ctx->host_stats.cluster_time);
                batch->start_timer();
                find_query_clusters(batch->vectors, ctx->centroids, batch->labels,
                                    batch->split_indices, batch->qc_num);
            }
            {
                ScopedTimer timer(ctx->host_stats.segment_time);
                compute_query_segment_scores(batch->vectors, ctx->centroids, batch->labels, ctx->cluster_var_list,
                                            batch->var_score_list, batch->qc_num, 0.5f, 0.5f);
            }
            
            queue_prepared.push(batch);
        }
        queue_prepared.shutdown();
    };

    // --- 线程 2: Reranker (后处理者) ---
    // 职责: 从 queue_for_rerank 获取已拉取结果的批次并进行后处理.
    auto reranker_logic = [&]() {
        std::shared_ptr<Batch> batch;
        int processed = 0;
        while (queue_for_rerank.pop(batch)) {
            // std::cout << "Reranker processing batch " << batch->offset << " (" << ++processed << "/" << total_batches << ")" << std::endl;
            {
                ScopedTimer timer(ctx->host_stats.rerank_time);
                rerank_and_store_topk(batch->results, ctx->topk_results, batch->size, batch->offset);
                batch->stop_timer();
                ctx->host_stats.add_batch_time(batch->get_processing_time());
            }

            if (++reranked_batches_count == total_batches) {
                // std::cout << "All batches processed by reranker" << std::endl;
                std::lock_guard<std::mutex> lock(final_sync_mutex);
                final_sync_cv.notify_one();
            }
        }
        // std::cout << "Reranker exiting, processed " << processed << " batches" << std::endl;
    };
    
    // --- 线程 3: IO_Manager (IO管理器) ---
    // 职责: 流水线的核心引擎, 严格串行化对MRAM总线的访问.
    auto io_manager_logic = [&]() {
        std::shared_ptr<Batch> pre_batch = nullptr;
        std::shared_ptr<Batch> curr_batch = nullptr;

        // 从队列中获取第一个batch作为启动
        if (queue_prepared.pop(curr_batch)) {
            // --- 流水线启动阶段 ---
            // 1. 广播第一个 batch 的数据并启动执行
            {
                ScopedTimer timer(ctx->host_stats.query_time);
                auto first_batch_launch = std::chrono::steady_clock::now();
                std::chrono::duration<double, std::micro> offset = first_batch_launch - ctx->start_time;
                ctx->host_prep_time_us = offset.count();
                broadcast_query_batch(ctx->system, curr_batch->vectors, curr_batch->labels, curr_batch->split_indices, 
                                      curr_batch->qc_num, curr_batch->var_score_list, curr_batch->offset);
            }
            // 2. 异步启动第一个 batch 的计算 (必须非阻塞)
            //    这里不再需要回调函数，因为同步点由后续的 sync() 控制
            async_system.call([](dpu::DpuSet&, unsigned){}, /*IsBlocking=*/false, /*SingleCall=*/true);
            
            pre_batch = curr_batch;
        }

        // --- 流水线主体循环 ---
        while (queue_prepared.pop(curr_batch)) {
            // 1. 等待上一个 batch (N-1) 的计算完成
            {
                ScopedTimer timer(ctx->host_stats.sync_time); // 统计同步时间
                async_system.sync(); // 这是关键的同步点
            }

            // 2. 拉取上一个 batch (N-1) 的结果
            {
                ScopedTimer timer(ctx->host_stats.pull_time);
                pull_query_result(ctx->system, pre_batch->results, ctx->dpu_stats, pre_batch->offset, pre_batch->size);
            }
            // 3. 将完成结果的 batch (N-1) 送去后处理
            queue_for_rerank.push(pre_batch);

            // 4. 广播当前 batch (N) 的数据并启动执行
            {
                ScopedTimer timer(ctx->host_stats.query_time);
                broadcast_query_batch(ctx->system, curr_batch->vectors, curr_batch->labels, curr_batch->split_indices, 
                                      curr_batch->qc_num, curr_batch->var_score_list, curr_batch->offset);
            }
            
            // 5. 注册当前 batch (N) 的回调
            async_system.call([](dpu::DpuSet&, unsigned){}, /*IsBlocking=*/false, /*SingleCall=*/true);
            
            // 6. 向前滚动流水线
            pre_batch = curr_batch;
        }

        // --- 流水线排空阶段 ---
        if (pre_batch != nullptr) {
            // 1. 等待最后一个 batch 的计算完成
            {
                ScopedTimer timer(ctx->host_stats.sync_time);
                async_system.sync();
            }
            // 2. 拉取最后一个 batch 的结果
            {
                ScopedTimer timer(ctx->host_stats.pull_time);
                pull_query_result(ctx->system, pre_batch->results, ctx->dpu_stats, pre_batch->offset, pre_batch->size);
            }
            // 3. 将最后一个 batch 送去后处理
            queue_for_rerank.push(pre_batch);
        }
        queue_for_rerank.shutdown(); // 所有IO和计算已完成
    };

    // --- 3. 启动并管理线程 ---
    std::thread producer_thread(producer_logic);
    std::thread io_manager_thread(io_manager_logic);
    std::thread reranker_thread(reranker_logic);
    
    // 等待所有线程完成它们的核心循环
    producer_thread.join();
    io_manager_thread.join();
    reranker_thread.join();

    async_system.sync();  // 确保所有DPU操作都已完成

    // 高效地等待最后一个批次被 Reranker 线程处理完毕
    {
        std::unique_lock<std::mutex> lock(final_sync_mutex);
        final_sync_cv.wait(lock, [&]{ return reranked_batches_count == total_batches; });
    }
}


void run_async_pipeline_query2(std::shared_ptr<QueryContext> ctx)
{
    // --- 1. 初始化 ---
    dpu::DpuSetAsync async_system = ctx->system.async(); 
    const int total_query_num = ctx->query_vectors.size();
    const int total_batches = (total_query_num + BATCH_SIZE - 1) / BATCH_SIZE;
    
    // 用于主线程等待所有任务完成的同步原语
    std::atomic<int> reranked_batches_count = 0;
    std::mutex final_sync_mutex;
    std::condition_variable final_sync_cv;

    std::atomic<int> dpu_callbacks_completed = 0;
    std::mutex callback_mutex;
    std::condition_variable callback_cv;

    // --- 2. 创建连接流水线各阶段的队列 ---
    ThreadSafeQueue<std::shared_ptr<Batch>> queue_prepared(PIPELINE_DEPTH);
    ThreadSafeQueue<std::shared_ptr<Batch>> queue_for_rerank(PIPELINE_DEPTH);

    // --- 线程 1: Producer (生产者) ---
    // 职责: 准备数据, 放入 queue_prepared.
    auto producer_logic = [&]() {
        for (int i = 0; i < total_batches; ++i) {
            int query_offset = i * BATCH_SIZE;
            const uint16_t batch_size = std::min(BATCH_SIZE, total_query_num - query_offset);

            auto batch = std::make_shared<Batch>();
            batch->allocate(batch_size, query_offset, ctx->query_vectors, ctx->qc_num, ctx->ground_vids);

            {
                ScopedTimer timer(ctx->host_stats.cluster_time);
                batch->start_timer();
                find_query_clusters(batch->vectors, ctx->centroids, batch->labels,
                                    batch->split_indices, batch->qc_num);
            }
            {
                ScopedTimer timer(ctx->host_stats.segment_time);
                compute_query_segment_scores(batch->vectors, ctx->centroids, batch->labels, ctx->cluster_var_list,
                                            batch->var_score_list, batch->qc_num, 0.5f, 0.5f);
                
                prepare_broadcast_buffer(batch);  // 准备分发数据的函数

            }
            queue_prepared.push(batch);
        }
        queue_prepared.shutdown();
    };

    // --- 线程 2: Reranker (后处理者) ---
    // 职责: 从 queue_for_rerank 获取已拉取结果的批次并进行后处理.
    auto reranker_logic = [&]() {
        std::shared_ptr<Batch> batch;
        int processed = 0;
        while (queue_for_rerank.pop(batch)) {
            // std::cout << "Reranker processing batch " << batch->offset << " (" << ++processed << "/" << total_batches << ")" << std::endl;
            {
                ScopedTimer timer(ctx->host_stats.rerank_time);
                process_results_and_rerank(ctx->dpu_stats, ctx->topk_results, batch);
                batch->stop_timer();
                ctx->host_stats.add_batch_time(batch->get_processing_time());
            }

            if (++reranked_batches_count == total_batches) {
                // std::cout << "All batches processed by reranker" << std::endl;
                std::lock_guard<std::mutex> lock(final_sync_mutex);
                final_sync_cv.notify_one();
            }
        }
        // std::cout << "Reranker exiting, processed " << processed << " batches" << std::endl;
    };
    
    // --- 线程 3: IO_Manager (IO管理器) ---
    // 职责: 流水线的核心引擎, 严格串行化对MRAM总线的访问.
    auto io_manager_logic = [&]() {
        std::shared_ptr<Batch> pre_batch = nullptr;
        std::shared_ptr<Batch> curr_batch = nullptr;

        // 从队列中获取第一个batch作为启动
        if (queue_prepared.pop(curr_batch)) {
            // 1. 广播第一个 batch 的数据并启动执行
            {
                ScopedTimer timer(ctx->host_stats.query_time);
                auto first_batch_launch = std::chrono::steady_clock::now();
                std::chrono::duration<double, std::micro> offset = first_batch_launch - ctx->start_time;
                ctx->host_prep_time_us = offset.count();
                simple_broadcast_query(ctx->system, ctx->dpu_stats, ctx->start_time, curr_batch->broadcast_buffer);
            }
            // 2. 异步启动第一个 batch 的计算 (必须非阻塞)
            //    这里不再需要回调函数，因为同步点由后续的 sync() 控制
            async_system.call([](dpu::DpuSet&, unsigned){}, /*IsBlocking=*/false, /*SingleCall=*/true);
            
            pre_batch = curr_batch;
        }

        // --- 流水线主体循环 ---
        while (queue_prepared.pop(curr_batch)) {
            // 1. 等待上一个 batch (N-1) 的计算完成
            {
                ScopedTimer timer(ctx->host_stats.sync_time); // 统计同步时间
                async_system.sync(); // 这是关键的同步点
            }
            // 2. 拉取上一个 batch (N-1) 的结果
            {
                ScopedTimer timer(ctx->host_stats.pull_time);
                simple_pull_results(ctx->system, pre_batch->pull_buffers);
            }
            // 3. 将完成结果的 batch (N-1) 送去后处理
            queue_for_rerank.push(pre_batch);

            // 4. 广播当前 batch (N) 的数据
            {
                ScopedTimer timer(ctx->host_stats.query_time);
                simple_broadcast_query(ctx->system, ctx->dpu_stats, ctx->start_time, curr_batch->broadcast_buffer);
            }
            
            // 5. 异步启动当前 batch (N) 的计算
            async_system.call([](dpu::DpuSet&, unsigned){}, /*IsBlocking=*/false, /*SingleCall=*/true);
            
            // 6. 向前滚动流水线
            pre_batch = curr_batch;
        }

        // --- 流水线排空阶段 ---
        if (pre_batch != nullptr) {
            // 1. 等待最后一个 batch 的计算完成
            {
                ScopedTimer timer(ctx->host_stats.sync_time);
                async_system.sync();
            }
            // 2. 拉取最后一个 batch 的结果
            {
                ScopedTimer timer(ctx->host_stats.pull_time);
                simple_pull_results(ctx->system, pre_batch->pull_buffers);
            }
            // 3. 将最后一个 batch 送去后处理
            queue_for_rerank.push(pre_batch);
        }
        queue_for_rerank.shutdown(); // 所有IO和计算已完成
    };

    // --- 3. 启动并管理线程 ---
    std::thread producer_thread(producer_logic);
    std::thread io_manager_thread(io_manager_logic);
    std::thread reranker_thread(reranker_logic);
    
    // 等待所有线程完成它们的核心循环
    producer_thread.join();
    io_manager_thread.join();
    reranker_thread.join();

    async_system.sync();  // 确保所有DPU操作都已完成

    // 高效地等待最后一个批次被 Reranker 线程处理完毕
    {
        std::unique_lock<std::mutex> lock(final_sync_mutex);
        final_sync_cv.wait(lock, [&]{ return reranked_batches_count == total_batches; });
    }
}

