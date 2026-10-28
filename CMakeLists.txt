cmake_minimum_required(VERSION 3.15)
project(PIM_ANNS CXX C)

# ========== 1. 配置参数 ==========

set(CMAKE_SKIP_RPATH OFF)  # 开启添加 RPATH 的选项
set(CMAKE_CXX_STANDARD 20)  # 要求 C++20 标准
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(NR_TASKLETS 16 CACHE STRING "Number of DPU tasklets")
message(STATUS "NR_TASKLETS default: ${NR_TASKLETS}")

set(QUERY_MODE 3 CACHE STRING "Query mode: 0 Sequential, 1 Pipeline, 2 Callback, 3 Async Pipeline")
set_property(CACHE QUERY_MODE PROPERTY STRINGS 0 1 2 3)
if (NOT "${QUERY_MODE}" MATCHES "^[0-3]$")
    message(FATAL_ERROR "QUERY_MODE must be one of 0, 1, 2, or 3.")
endif()
message(STATUS "Using query mode: ${QUERY_MODE}")


# 使用安装在 /usr 下的系统 UPMEM SDK
set(SDK_INSTALL_DIR "/usr")


set(DPU_COMPILER ${SDK_INSTALL_DIR}/bin/dpu-upmem-dpurte-clang)  # 编译DPU程序使用的clang
message(STATUS "Using corresponding DPU Compiler: ${DPU_COMPILER}")

set(SDK_LIB_DIR "${SDK_INSTALL_DIR}/lib")
set(FAISS_LIB_DIR "/usr/local/lib")  # Faiss 相关库
set(FAISS_INCLUDE_DIR "/usr/local/include/faiss")

# 写入 RPATH，避免手动 export 库
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${SDK_LIB_DIR}")

# build输出目录
set(BUILDDIR "${CMAKE_BINARY_DIR}")
file(MAKE_DIRECTORY ${BUILDDIR})

# ========== 2. 源文件收集 ==========
# (1) 收集 host/ 顶层的源文件
set(HOST_MAIN_SOURCES
    host/main.cc
    host/config.cc
    host/utils.cc
    host/stats.cc
)

# (2) 收集 host 查询和数据传输文件
set(HOST_QUERY_SOURCE  
    ${CMAKE_CURRENT_SOURCE_DIR}/host/host_process/query_batch.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/host/host_process/dpu_host_xfer.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/host/host_process/host_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/host/host_process/query_batch_cpu.cc
)

# 最后收集有关数据处理的文件
set(HOST_DATA_UTILS_SOURCES
    host/data_utils/dataset.cc
    host/data_utils/kmeans.cc
    host/data_utils/cluster_var.cc
)

# (3) 将所有 host 源文件合并到一个变量中
set(HOST_SOURCES ${HOST_MAIN_SOURCES} ${HOST_DATA_UTILS_SOURCES} ${HOST_QUERY_SOURCE})

# (4) SHARE 部分使用默认搜集所有文件
file(GLOB SHARE_HEADERS "share/*.h")
file(GLOB SHARE_SOURCES "share/*.c")

add_compile_definitions(QUERY_MODE=${QUERY_MODE}) # 将 QUERY_MODE 传递给代码（C/C++）


# (6) 收集 DPU 批量查询的源文件和头文件
file(GLOB DPU_HEADERS "dpu/*.h" "dpu/dpu_process/*.h")
set(DPU_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/dpu/dpu_process/dpu_batch_kernel.c
    ${CMAKE_CURRENT_SOURCE_DIR}/dpu/dpu_process/dpu_base.c
    ${CMAKE_CURRENT_SOURCE_DIR}/dpu/dpu_main.c
    ${CMAKE_CURRENT_SOURCE_DIR}/dpu/dpu_utils.c
    ${CMAKE_CURRENT_SOURCE_DIR}/dpu/heap.c
    ${CMAKE_CURRENT_SOURCE_DIR}/dpu/dpu_global.c
)

set(HOST_BINARY ${BUILDDIR}/host_app)
set(DPU_BINARY ${BUILDDIR}/dpu_task)

# ========== 3. DPU程序 ==========

# DPU编译，不用add_executable，因为要用特殊编译器
# 注意对于 spacev100m 数据集，在查询 cluster 多的时候，必须把stack和seqreader空间开大
add_custom_command(
    OUTPUT ${DPU_BINARY}
    COMMAND ${DPU_COMPILER}
        -g -O3
        -DNR_TASKLETS=${NR_TASKLETS}
        -DQUERY_MODE=${QUERY_MODE}
        -DSTACK_SIZE_DEFAULT=2048
        -DSEQREAD_CACHE_SIZE=256
        -Wno-ignored-attributes
        -I${SDK_INSTALL_DIR}/share/upmem/include
        ${DPU_SOURCES} ${SHARE_SOURCES}
        -o ${DPU_BINARY}
    DEPENDS ${DPU_SOURCES} ${DPU_HEADERS} ${SHARE_SOURCES} ${SHARE_HEADERS}
    COMMENT "Building DPU binary"
)

add_custom_target(build_dpu ALL DEPENDS ${DPU_BINARY})

# ========== 4. Host应用 ==========

# 1. 定义可执行文件和源文件
add_executable(host_app ${HOST_SOURCES} ${SHARE_SOURCES})

# 2. 设置编译时属性 (头文件、宏定义、编译选项)
target_include_directories(host_app PRIVATE
    ${FAISS_INCLUDE_DIR}
    ${SDK_INSTALL_DIR}/include/dpu
)

target_compile_definitions(host_app PRIVATE
    NR_TASKLETS=${NR_TASKLETS}
    QUERY_MODE=${QUERY_MODE}
    DPU_BINARY=\"${DPU_BINARY}\"
)

# 使用SIMD指令优化host一侧nprobe的查询，需要增加编译选项 -mavx2 -mfma
target_compile_options(host_app PRIVATE
    -g -O3 -pthread -mavx2 -mfma
)

# 3. 设置链接时属性 (库搜索路径、RPATH)
#    把 RPATH 设置放在链接库之前！
target_link_directories(host_app PRIVATE
    ${SDK_LIB_DIR}
    ${FAISS_LIB_DIR}
)

target_link_options(host_app PRIVATE "-Wl,-rpath,${SDK_INSTALL_DIR}/lib")

# 4. 链接具体的库 (通常是最后一步)
find_package(OpenMP REQUIRED)
target_link_libraries(host_app
    PRIVATE # 建议加上 PRIVATE 或 PUBLIC 关键字
    dpu
    faiss
    OpenMP::OpenMP_CXX
)

# 5. (可选) 对于安装的属性设置可以保留，但它主要影响 make install
set_target_properties(host_app PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
)

# ========== 5. 安装、clean等（可选） ==========

# install(TARGETS host_app DESTINATION bin)
# install(FILES ${DPU_BINARY} DESTINATION bin)
