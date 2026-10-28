#ifndef FLAT_MATRIX_HH
#define FLAT_MATRIX_HH

#include <vector>
#include <cstdint>
#include <stdexcept>
#include <cstring>

// C++20's std::span is a non-owning view of a contiguous sequence of objects.
// It's the perfect, safe way to return a view of a vector from our flat memory.
#if __cplusplus >= 202002L
#include <span>
#endif


// 二维规则数组的展平类
template<typename T>
struct FlatMatrix {
    std::vector<T> data;  // 存储所有向量数据的单一、连续内存块
    size_t num_vectors = 0;  // 向量的数量 (行数)
    size_t dimension = 0;  // 每个向量的维度 (列数)

    // 默认构造函数 (创建 0 x 0 的矩阵)
    FlatMatrix() = default;

    // 带参数的构造函数，预留空间
    FlatMatrix(size_t n_vectors, size_t dim) : num_vectors(n_vectors), dimension(dim) {
        data.resize(num_vectors * dimension);
    }

    // 重置并填充已存在的矩阵
    void assign(size_t n_vectors, size_t dim, const T& fill_value) {
        num_vectors = n_vectors;
        dimension = dim;
        data.assign(n_vectors * dim, fill_value);
    }
    
    // 给矩阵预留空间
    void resize(size_t n_vectors, size_t dim) {
        num_vectors = n_vectors;
        dimension = dim;
        data.resize(n_vectors * dim);
    }

    // 清空数据
    void clear() {
        data.clear();
        num_vectors = 0;
        dimension = 0;
    }

    // 获取向量总数
    size_t size() const {
        return num_vectors;
    }

    // 获取维度
    size_t dim() const {
        return dimension;
    }

    // 裁剪 cols （用于裁剪 TOP_K ）
    void trim_cols(size_t new_dim) {
        // printf("new_dim:%ld, orginal dimension:%ld\n", new_dim, dimension);
        if (new_dim > dimension)
            throw std::invalid_argument("Cannot increase dimension.");

        if (new_dim == dimension) return;

        std::vector<T> new_data;
        new_data.resize(num_vectors * new_dim);
        for (size_t i = 0; i < num_vectors; ++i) {
            std::memcpy(&new_data[i * new_dim], &data[i * dimension], new_dim * sizeof(T));
        }
        data = std::move(new_data);
        dimension = new_dim;
    }

    // 裁剪 rows （用于裁剪 query）
    void trim_rows(size_t new_n) {
        // printf("new_n:%ld, num_vectors:%ld\n", new_n, num_vectors);
        if (new_n > num_vectors)
            throw std::invalid_argument("Cannot increase row count.");
        data.resize(new_n * dimension);
        num_vectors = new_n;
    }

    // 将二维 std::vector 转换为 FlatMatrix
    static FlatMatrix<T> create_from(const std::vector<std::vector<T>>& vec2d) {
        FlatMatrix<T> matrix;
        if (vec2d.empty()) {
            throw std::runtime_error("Input vector of vectors is none!");
            return matrix; // 如果输入为空，返回一个空矩阵
        }

        matrix.num_vectors = vec2d.size();
        matrix.dimension = vec2d[0].size(); // 以第一行为基准维度
        matrix.data.resize(matrix.num_vectors * matrix.dimension);

        #pragma omp parallel for
        for (size_t i = 0; i < matrix.num_vectors; ++i) {
            // 安全检查：确保输入是规则的矩阵（所有行维度相同）
            if (vec2d[i].size() != matrix.dimension) {
                throw std::runtime_error("Input vector of vectors is not a regular matrix.");
            }
            T* dest_row = matrix.data.data() + i * matrix.dimension;
            std::memcpy(dest_row, vec2d[i].data(), matrix.dimension * sizeof(T));
        }
        return matrix;
    }

#if __cplusplus >= 202002L
    // 加上 always_inline 关键字，并用宏强制内联
    inline __attribute__((always_inline))
    std::span<const T> operator[](size_t i) const {
        // 只在Debug模式下进行边界检查
        // #ifndef NDEBUG
        // if (i >= num_vectors) throw std::out_of_range("Vector index out of range.");
        // #endif
        return {data.data() + i * dimension, dimension};
    }

    inline __attribute__((always_inline))
    std::span<T> operator[](size_t i) {
        // #ifndef NDEBUG
        // if (i >= num_vectors) throw std::out_of_range("Vector index out of range.");
        // #endif
        return {data.data() + i * dimension, dimension};
    }
#else
    // 在 C++17 或更早版本的环境下 (返回裸指针)
    inline __attribute__((always_inline))
    const T* operator[](size_t i) const {
        // #ifndef NDEBUG
        // if (i >= num_vectors) throw std::out_of_range("Vector index out of range.");
        // #endif
        return data.data() + i * dimension;
    }

    inline __attribute__((always_inline))
    T* operator[](size_t i) {
        // #ifndef NDEBUG
        // if (i >= num_vectors) throw std::out_of_range("Vector index out of range.");
        // #endif
        return data.data() + i * dimension;
    }
#endif
};


// 二维不规则数组的展平类
template<typename T>
struct JaggedArray {
    std::vector<T> data;         // 所有子数组的元素拼接在一起
    std::vector<size_t> offsets; // 每个子数组在 data 中的起始索引

    // 返回子数组的数量
    size_t size() const {
        return offsets.empty() ? 0 : offsets.size() - 1;
    }

    // 将二维 std::vector 转换为 JaggedArray
    static JaggedArray<T> create_from(const std::vector<std::vector<T>>& vec2d) {
        JaggedArray<T> jagged;
        size_t num_rows = vec2d.size();
        jagged.offsets.resize(num_rows + 1, 0);

        // 预计算总元素数，一次性分配 data 内存
        size_t total_elements = 0;
        for (size_t i = 0; i < num_rows; ++i) {
            total_elements += vec2d[i].size();
        }
        jagged.data.reserve(total_elements);

        // 填充 data 和 offsets
        for (size_t i = 0; i < num_rows; ++i) {
            jagged.data.insert(jagged.data.end(), vec2d[i].begin(), vec2d[i].end());
            jagged.offsets[i + 1] = jagged.data.size();
        }
        return jagged;
    }

#if __cplusplus >= 202002L
    // 通过索引访问第 i 个子数组 (返回一个视图，无拷贝)
    inline __attribute__((always_inline))
    std::span<const T> operator[](size_t i) const {
        // #ifndef NDEBUG
        // if (i >= count()) throw std::out_of_range("JaggedArray index out of range.");
        // #endif
        const size_t start_offset = offsets[i];
        const size_t end_offset = offsets[i+1];
        return {data.data() + start_offset, end_offset - start_offset};
    }
#else
    // C++17 或更早版本的替代方案，返回一个指针和长度的 pair
    std::pair<const T*, size_t> operator[](size_t i) const {
        // if (i >= size()) throw std::out_of_range("JaggedArray index out of range.");
        const size_t start_offset = offsets[i];
        const size_t end_offset = offsets[i+1];
        return {data.data() + start_offset, end_offset - start_offset};
    }
#endif
};


#endif // FLAT_MATRIX_HH