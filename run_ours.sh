#!/bin/bash

# ==============================================================================
#  脚本说明 (README)
# ==============================================================================
#
#  本脚本用于执行一组"Our Method"实验。
#     - 将 IS_BASELINE 设为 false。
#     - 遍历 DIST_TYPE (0, 1, 2)。
#     - 文件名中不含额外后缀。
#     - 遍历指定的 BATCH_SIZE 和数据集，程序内部完成 nprobe sweep。
#
# ==============================================================================

# --- 配置区 ---
# 命令出错时立即退出
set -e

# (1) 文件路径
HOST_UTILS_HH="./host/utils.hh"
SHARE_GLOBAL_H="./share/global.h"
DPU_GLOBAL_H="./dpu/dpu_global.h"

# (2) 参数数组
DATA_NAME_VALS=(0 1)  # 0:sift, 1:spacev
BATCH_SIZES=(128)  # 调大可以提高吞吐量，但同时会增加latency
QUERY_TOP_K=100  
P_DEPTH=1  # 流水线深度，默认为1。增大可以提高吞吐量，但同时会增加latency

# (3) 构建配置
CMAKE_CMD="$HOME/cmake-3.31.6/bin/cmake"
BUILD_DIR="./build"
EXECUTABLE_PATH="${BUILD_DIR}/host_app"
RESULTS_DIR="./bash_qps_results_top${QUERY_TOP_K}"
DPU_LOG_DIR="./bash_qps_dpu_counts_top${QUERY_TOP_K}"

# Our Method 的参数
QUERY_MODES_OURS=(3)  # 3 为host采用async pipeline的方法；0 为无pipeline情况，搭配DIST_TYPE=0使用
DIST_TYPE_OURS=(1)  # 可选 0 1 2；其中2是分段阈值约束的，暂时不用
QUERY_METHOD_DPU="MyQuery"


# TOP_K 合法性检查
if [ "$QUERY_TOP_K" -ne 10 ] && [ "$QUERY_TOP_K" -ne 100 ]; then
    echo "TOP_K: $QUERY_TOP_K not supported!"
    exit 1
fi


# --- 脚本主体 ---

# 确保源代码文件在脚本退出时恢复原样
# MODIFIED_FILES=(
#     "${HOST_UTILS_HH}"
#     "${SHARE_GLOBAL_H}"
#     "${DPU_GLOBAL_H}"
# )
# echo "保存当前文件状态..."
# trap 'echo "恢复源代码文件..."; git checkout -- "${MODIFIED_FILES[@]}"' EXIT

# 创建结果输出目录
mkdir -p "${RESULTS_DIR}"
mkdir -p "${DPU_LOG_DIR}"


# ==============================================================================
#                              OUR METHOD 测试
# ==============================================================================
echo " --- 开始 OUR METHOD 实验 --- "

# 我们方法的文件名不带后缀

# --- 激活 Our Method 模式 ---
echo "--> 激活 Our Method 模式 (IS_BASELINE = false), TOP_K: ${QUERY_TOP_K}, PIPELINE_DEPTH: ${P_DEPTH}"
sed -i "s/#define IS_BASELINE \(true\|false\)/#define IS_BASELINE false/" "${SHARE_GLOBAL_H}"
sed -i "s|#define TOP_K [0-9]\+|#define TOP_K ${QUERY_TOP_K}|" "${SHARE_GLOBAL_H}"
sed -i "s/constexpr int PIPELINE_DEPTH = [0-9]\+;/constexpr int PIPELINE_DEPTH = ${P_DEPTH};/" "${HOST_UTILS_HH}"

# Our Method 的循环
for q_mode in "${QUERY_MODES_OURS[@]}"; do
    echo "--> 设置 QUERY_MODE = ${q_mode}"
    for data_val in "${DATA_NAME_VALS[@]}"; do
        if [ "${data_val}" -eq 0 ]; then
            data_str="sift"
        else
            data_str="spacev"
        fi

        echo "--> 选择目标数据集：${data_str}"
        sed -i "s/#define DATA_NAME [0-9]\+/#define DATA_NAME ${data_val}/" "${SHARE_GLOBAL_H}"

        for bs in "${BATCH_SIZES[@]}"; do
            sed -i "s/inline constexpr int BATCH_SIZE = [0-9]\+;/inline constexpr int BATCH_SIZE = ${bs};/" "${HOST_UTILS_HH}"
            
            for dist in "${DIST_TYPE_OURS[@]}"; do
                # 在 #else 块中修改 L2_DIST_METHOD
                echo "--> 设置 L2_DIST_METHOD = ${dist}"
                sed -i "/#else/,/#endif/s/#define L2_DIST_METHOD [0-9]\+/#define L2_DIST_METHOD ${dist}/" "${DPU_GLOBAL_H}"

                output_filename="${RESULTS_DIR}/${data_str}_QM${q_mode}_BS${bs}_DEPTH${P_DEPTH}_DTYPE${dist}.log"
                dpu_log_filename="${DPU_LOG_DIR}/dpu_number_${data_str}_QM${q_mode}_BS${bs}_DEPTH${P_DEPTH}_DTYPE${dist}.log"

                echo ""
                echo "***> [OURS] 测试: DATA=${data_str}, QM=${q_mode}, C=sweep, BS=${bs}, DTYPE=${dist}, P_DEPTH=${P_DEPTH}"

                echo "--> 正在编译..."
                "${CMAKE_CMD}" -DQUERY_MODE:INT=${q_mode} -S . -B "${BUILD_DIR}" > /dev/null
                make -C "${BUILD_DIR}" -j$(nproc)

                echo "--> 正在运行 nprobe sweep..."
                "${EXECUTABLE_PATH}" --log_path "${dpu_log_filename}" --query_method "${QUERY_METHOD_DPU}" > "${output_filename}" 2>&1
            done
        done
    done
done

echo ""
echo "=================================================================="
echo "所有实验已完成！结果保存在 ${RESULTS_DIR} 目录中。"
exit 0
