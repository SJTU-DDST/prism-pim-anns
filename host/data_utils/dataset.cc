#include "dataset.hh"

Dataset load_dataset(const Config& config) 
{
    std::string data_path = config.data_path;
    bool query_debug = config.query_debug;
    int query_size = config.query_size;

    const std::string dataset_name = get_dataset_name();
    int data_size = get_dataset_size();

    // get dataset file name for siftsmall (sift10k) and sift (sift1m)
    std::string base_file = data_path + dataset_name + "/" + dataset_name + "_base.fvecs";
    std::string query_file = data_path + dataset_name + "/" + dataset_name + "_query.fvecs";
    std::string ground_truth_file = data_path + dataset_name + "/" + dataset_name + "_groundtruth.ivecs";
    if ((DATA_NAME == 0) && (DATA_SIZE == 3 or DATA_SIZE == 4)) {  // sift10m and sift 100m
        base_file = data_path + dataset_name + "/" + dataset_name + "_base.bvecs";
        query_file = data_path + dataset_name + "/" + dataset_name + "_query.bvecs";
        ground_truth_file = data_path + dataset_name + "/gnd/idx_" + std::to_string(data_size) + "M.ivecs";
    }
    else if ((DATA_NAME == 1) && (DATA_SIZE == 1)) {  // spacev10k
        base_file = data_path + dataset_name + "/" + dataset_name + "10k.bin";
        query_file = data_path + dataset_name + "/" +"query_spacev.bin";
        ground_truth_file = data_path + dataset_name + "/" + dataset_name + "10k_gt100.bin";
    }
    else if ((DATA_NAME == 1) && (DATA_SIZE == 2 or DATA_SIZE == 3 or DATA_SIZE == 4)) {  // spacev1m, 10m, and 100m
        base_file = data_path + dataset_name + "/" + dataset_name + std::to_string(data_size) + "m.bin";
        query_file = data_path + dataset_name + "/" +"query_spacev.bin";
        ground_truth_file = data_path + dataset_name + "/" + dataset_name + std::to_string(data_size) + "m_gt100.bin";
    }

    if (!std::filesystem::exists(base_file)) {
        throw std::runtime_error("Base vector file not found: " + base_file);
    }
    if (!std::filesystem::exists(query_file)) {
        throw std::runtime_error("Query vector file not found: " + query_file);
    }
    if (!std::filesystem::exists(ground_truth_file)) {
        throw std::runtime_error("Ground truth vector id file not found: " + ground_truth_file);
    }

    std::cout << "[Load] Read vectors from: " << base_file << std::endl;
    Dataset dataset;
    dataset.dataset_name = dataset_name;
    #if (DATA_NAME == 0) && ((DATA_SIZE == 1) || (DATA_SIZE == 2))
        FlatMatrix<float> base_vectors_f = read_fvecs(base_file);
        FlatMatrix<float> query_vectors_f = read_fvecs(query_file);
        // sift 数据集的 base_vector 的每个维度的取值范围是 [0, 255] (uint8_t)，可直接转换
        convert_vectors_to_uint8(base_vectors_f, dataset.base_vectors);
        convert_vectors_to_uint8(query_vectors_f, dataset.query_vectors);
        dataset.ground_truth_vids = read_ivecs(ground_truth_file);
    #elif (DATA_NAME == 0) && ((DATA_SIZE == 3) || (DATA_SIZE == 4))
        dataset.base_vectors = read_bvecs(base_file, data_size * 1'000'000);
        dataset.query_vectors = read_bvecs(query_file);
        dataset.ground_truth_vids = read_ivecs(ground_truth_file);
    #elif (DATA_NAME == 1)
        std::vector<std::vector<float>> truth_dists;
        dataset.base_vectors = read_raw_data_bin(base_file);
        dataset.query_vectors = read_raw_data_bin(query_file);
        auto [gt_vids, gt_dists] = read_ground_truth_bin(ground_truth_file); 
        dataset.ground_truth_vids = std::move(gt_vids);
    #endif

    if (((DATA_NAME == 0) && ((DATA_SIZE == 3) || (DATA_SIZE == 4)) && TOP_K < 1000) 
        || (DATA_NAME == 1) && TOP_K < 100) {
        std::cout << "[Small TOP_K] Trimming ground truth topK to " << TOP_K << ".\n";
        dataset.ground_truth_vids.trim_cols(TOP_K);
    }

    if (query_debug) {
        if (query_size < dataset.query_vectors.size()) {
            std::cout << "[Debug] Trimming query number to " << query_size << " vectors for debugging.\n";
            dataset.ground_truth_vids.trim_rows(query_size);
            dataset.query_vectors.trim_rows(query_size);
        } else {
            std::cout << "[No Debug] Input query number (" << query_size 
                      << ") larger then dataset query_size (" << dataset.query_vectors.size() 
                      << "), will use whole query vectors.\n";
        }
    }

    return dataset;
}


FlatMatrix<float> read_fvecs(const std::string& filename) 
{
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open file: " + filename + "for reading.");
    }

    FlatMatrix<float> matrix;
    uint32_t dim;

    // 读取第一个向量来确定维度和计算记录大小
    if (!in.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t))) {
        return matrix; // 文件为空
    }
    matrix.dimension = dim;
    const size_t record_size = sizeof(uint32_t) + dim * sizeof(float);

    // 计算文件中的向量总数
    in.seekg(0, std::ios::end);
    matrix.num_vectors = static_cast<size_t>(in.tellg()) / record_size;
    in.seekg(0, std::ios::beg);
    matrix.data.resize(matrix.num_vectors * matrix.dimension);  // 一次性分配所有内存

    // 循环读取，只拷贝向量数据
    for (size_t i = 0; i < matrix.num_vectors; ++i) {
        in.seekg(sizeof(uint32_t), std::ios::cur); // 跳过维度信息
        in.read(reinterpret_cast<char*>(matrix.data.data() + i * matrix.dimension),
                matrix.dimension * sizeof(float));
    }
    
    return matrix;
}


FlatMatrix<int> read_ivecs(const std::string& filename) 
{
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open file: " + filename + "for reading.");
    }

    FlatMatrix<int> matrix;
    uint32_t dim;
    // 读取第一个向量来确定维度和计算记录大小
    if (!in.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t))) {
        return matrix; // 文件为空
    }
    
    matrix.dimension = dim;
    const size_t record_size = sizeof(uint32_t) + dim * sizeof(int);

    // 计算文件中的向量总数
    in.seekg(0, std::ios::end);
    matrix.num_vectors = static_cast<size_t>(in.tellg()) / record_size;
    in.seekg(0, std::ios::beg);
    matrix.data.resize(matrix.num_vectors * matrix.dimension);  // 一次性分配所有内存

    // 循环读取，只拷贝向量数据
    for (size_t i = 0; i < matrix.num_vectors; ++i) {
        in.seekg(sizeof(uint32_t), std::ios::cur); // 跳过维度信息
        in.read(reinterpret_cast<char*>(matrix.data.data() + i * matrix.dimension),
                matrix.dimension * sizeof(float));
    }
    return matrix;
}

// 版本1：读取全部数据（无data_size参数）
FlatMatrix<uint8_t> read_bvecs(const std::string& filename) 
{
    return read_bvecs(filename, -1); // 调用完整版，-1表示读入全部数据
}

// 版本2：带read_size参数控制读取量（int类型，-1表示全部）
FlatMatrix<uint8_t> read_bvecs(const std::string& filename, int read_size) 
{
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open file: " + filename);

    FlatMatrix<uint8_t> matrix;
    uint32_t dim;

    // 读取维度并计算记录大小
    if (!in.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t))) {
        return matrix; // 文件为空
    }
    matrix.dimension = dim;
    const size_t record_size = sizeof(uint32_t) + dim;
    in.seekg(0, std::ios::beg); // 重置回文件开头

    // 计算要读取的向量数
    size_t num_to_reserve = 0;
    #if (DATA_NAME == 0) // 针对于 bigann 数据集
        if (read_size == -1) {
            num_to_reserve = 10000;  // 默认是 query 的时候，不传递数据集的大小，预留 10000
        } else {
            #if (DATA_SIZE == 1)
                num_to_reserve = 10'000;
            #elif (DATA_SIZE == 2)
                num_to_reserve = 1'000'000;
            #elif (DATA_SIZE == 3)
                num_to_reserve = 10'000'000;
            #elif (DATA_SIZE == 4)
                num_to_reserve = 100'000'000;
            #else
                // 如果没有宏，从文件大小计算
                in.seekg(0, std::ios::end);
                num_to_reserve = static_cast<size_t>(in.tellg()) / record_size;
                in.seekg(0, std::ios::beg);
            #endif
        }
    #endif
    matrix.num_vectors = num_to_reserve;
    
    // 一次性分配内存
    matrix.data.resize(matrix.num_vectors * matrix.dimension);

    // 循环读取
    std::cout << "[Load] Reading " << matrix.num_vectors << " vectors...\n";
    for (size_t i = 0; i < matrix.num_vectors; ++i) {
        // 读取并丢弃维度信息
        uint32_t temp_dim;
        if (!in.read(reinterpret_cast<char*>(&temp_dim), sizeof(uint32_t))) {
            matrix.num_vectors = i; // 文件提前结束，修正向量数
            matrix.data.resize(i * matrix.dimension);
            break;
        }

        // 直接读取到扁平内存的正确位置
        if (!in.read(reinterpret_cast<char*>(matrix.data.data() + i * matrix.dimension), matrix.dimension)) {
            matrix.num_vectors = i; // 文件提前结束
            matrix.data.resize(i * matrix.dimension);
            break;
        }
        
        if ((i + 1) % 10'000'000 == 0) {
            std::cout << "[Load progress] Loaded " << (i + 1) / 1'000'000 << "M vectors\n";
        }
    }
    
    std::cout << "[Load done] Loaded " << matrix.size() << " vectors.\n";
    return matrix;
}


FlatMatrix<int8_t> read_raw_data_bin(const std::string& filename) 
{
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
        
    int32_t v_count, dimension;  // 读取元数据
    in.read(reinterpret_cast<char*>(&v_count), sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&dimension), sizeof(int32_t));

    FlatMatrix<int8_t> matrix(v_count, dimension);  // 初始化矩阵
    const size_t total_bytes = matrix.num_vectors * matrix.dimension * sizeof(int8_t);
    
    // 一次I/O调用读取所有数据
    if (!in.read(reinterpret_cast<char*>(matrix.data.data()), total_bytes)) {
        throw std::runtime_error("Error reading vector data from file or file is truncated.");
    }
    
    return matrix;
}


std::pair<FlatMatrix<int32_t>, FlatMatrix<float>> 
read_ground_truth_bin(const std::string& filename) 
{
    std::ifstream ftruth(filename, std::ios::binary);
    if (!ftruth) {
        throw std::runtime_error("Cannot open ground truth file: " + filename);
    }

    // 读取头信息: 向量总数和每个向量的近邻数(维度)
    int32_t t_count, topk;
    if (!ftruth.read(reinterpret_cast<char*>(&t_count), sizeof(int32_t)) ||
        !ftruth.read(reinterpret_cast<char*>(&topk), sizeof(int32_t))) {
        // 如果连头信息都读不完整，抛出异常
        throw std::runtime_error("Could not read header from ground truth file: " + filename);
    }
    
    if ((t_count <= 0) || (topk <= 0)) {
        // 如果文件头数据异常
        throw std::runtime_error("Header error from ground truth file: " + filename);
    }

    FlatMatrix<int32_t> vids_matrix(t_count, topk);
    FlatMatrix<float> dists_matrix;  // dists 暂时不需要，因此初始化空矩阵，也不读取

    const size_t vids_bytes = static_cast<size_t>(t_count) * topk * sizeof(int32_t);

    // 一次性读完整个的 dists 数据块
    if (!ftruth.read(reinterpret_cast<char*>(vids_matrix.data.data()), vids_bytes)) {
        throw std::runtime_error("Error reading ground truth IDs from file or file is truncated: " + filename);
    }

    return {vids_matrix, dists_matrix};
}


void convert_vectors_to_uint8(const FlatMatrix<float>& source, 
                              FlatMatrix<uint8_t>& destination)
{
    // 完成目标矩阵的尺寸设定和内存分配
    destination.resize(source.num_vectors, source.dimension);

    // 获取指向底层连续内存的裸指针
    const float* src_ptr = source.data.data();
    uint8_t* dest_ptr = destination.data.data();

    const size_t total_elements = source.data.size();
    for (size_t i = 0; i < total_elements; ++i) {
        dest_ptr[i] = static_cast<uint8_t>(src_ptr[i]);  // 直接转换
    }
}
