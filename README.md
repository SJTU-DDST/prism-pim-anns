# Prism: Reducing Arithmetic for High-Recall Approximate Nearest Neighbor Search on Processing-in-Memory

**简体中文** | [English](README.en.md)

Prism 是一个基于 PIM 的 ANNS 系统，协同优化向量剪枝、高维距离计算以及host-PIM协同。
它采用邻近感知的向量剪枝器，利用 PIM 内部高带宽和双簇关联来过滤掉距离较远的向量。
随后，Prism 执行按敏感度排序的距离计算，优先处理高影响维度段，并在满足排除条件时提前终止候选向量的距离计算。
无停滞的host-PIM 流水线将查询准备、PIM 执行和全局排序重叠进行，以维持高吞吐量。
在UPMEM PIM的实验平台上，Prism 相比最先进的系统，吞吐量达到 2.7–19.8 倍。


## 代码文件说明
默认的数据集路径为`/mnt/anns-dataset`。`./dpu`存放DPU一侧的代码，`./host`存放host一侧的代码，`./share`存放host和DPU共享的变量、宏定义等。`./plot`是绘图的python代码。


## 编译依赖
项目使用安装在 `/usr` 下的系统 UPMEM SDK，包括 DPU 编译器、host 头文件和 `libdpu`。编译还需要 CMake 3.15 或更新版本、支持 C++20 的编译器、OpenMP 和 Faiss。


# 编译和运行代码
## 项目代码依赖库faiss的安装
1. 本项目的C++代码依赖faiss库进行聚类，需要从源码编译和安装。
首先安装构建所需的依赖库：`sudo apt-get install libgflags-dev libblas-dev liblapack-dev swig`
2. 从Github下载或者clone faiss源代码，解压进入目录，若通过 ZIP 压缩包获取源码，要把faiss-main重命名成faiss。
创建build目录：
```mkdir build```
设置编译参数（我们UPMEM主机的CPU是Intel Xeon Silver 4216 CPU @ 2.10GHz，支持Intel BLAS，不支持avx512_spr）：
```
cmake -B build . \
    -DFAISS_ENABLE_GPU=OFF \
    -DBUILD_TESTING=OFF \
    -DFAISS_OPT_LEVEL=avx512 \
    -DCMAKE_BUILD_TYPE=Release \
    -DBLA_VENDOR=Intel10_64_dyn
```
编译：
```make -C build -j```
安装：
```sudo make -C build install```
注意，在启用了Intel MKL的情况下，可能编译出来的库无法找到MKL库（用ldd查看编译的文件，出现`libmkl_rt.so.2 => not found`），此时需要
```
echo 'export LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/2025.2/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```
来把MKL安装的路径加入到用户自己的环境变量中。
3. 在解压的faiss源码的faiss目录下，impl文件夹要复制到/usr/local下，否则某些include可能找不到。具体为`sudo cp -r ~/faiss/faiss/impl /usr/local/include/faiss/impl`


## 编译和运行代码
1. 在项目根目录配置并编译：
```bash
cmake -S . -B build -DQUERY_MODE=3 -DNR_TASKLETS=16
cmake --build build -j
```
2. 在项目根目录运行，部分文件以相对路径寻址：
```bash
./build/host_app
```
3. 在我们的 UPMEM 服务器上，每次它重启后，默认的CPU为省电模式（powersave），可通过命令`cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`查看电源模式，并通过`echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`来修改到性能模式。


## 代码参数位置
1. host和DPU共享参数主要在 `./share/global.h` 中（例如数据集选择、数据集大小），DPU参数主要在 `./dpu/dpu_global.c` 中（例如DPU内的优化方法），host参数主要在 `./host/config.cc` 中（例如查询方法），有关 batch 的参数放在 `./host/utils.hh` 中。
2. 通过 CMake 参数设置 DPU 线程数量和查询模式，例如 `cmake -S . -B build -DNR_TASKLETS=16 -DQUERY_MODE=3`。`QUERY_MODE` 支持 0（串行）、1（串行流水线）、2（异步回调）和 3（异步流水线），默认为 3。重新配置后编译即可生效。
3. 可以通过 bash 脚本来运行代码。`run_baseline.sh`、`run_cpu_baseline.sh` 和 `run_ours.sh` 对每组数据集、查询模式和 batch 等配置只启动一次程序。100M 数据集的 nprobe 参数集中在 `host/utils.cc` 的 `get_qc_num_sweep()` 中，按 `DATA_NAME`、`TOP_K` 和查询方案选择；CPU baseline 使用独立参数表。
4. 命令行参数由 cxxopts 管理，可运行 `./build/host_app --help` 查看数据路径、查询方法和查询数量等选项。


## 运行各种方案的参数
1. 运行DPU baseline，需要指定 `./share/global.h` 中的宏 `MIN_REPLICA` 和 `MAX_REPLICA` 为1，指定 `./dpu/dpu_global.h` 中的宏 `L2_DIST_METHOD` 为0。或者运行脚本 `./run_dpu_baseline.sh`
2. 启用Host-PIM Stall-Free Execution Pipeline，在编译时使用参数 `-DQUERY_MODE=3` 。
3. 运行Intersection-Based Vector Pruning，需要指定 `./share/global.h` 中的宏 `MIN_REPLICA` 和 `MAX_REPLICA` 为2，此时交叉查询区域的宏定义 `IS_CROSS_QUERY` 会自动设置为true。
4. 运行Sensitivity-Ordered Distance Computation，需要指定 `./dpu/dpu_global.h` 中的宏 `L2_DIST_METHOD` 为1。
5. 将上述2-4结合即为Prism方案，可以直接运行脚本 `./run_ours.sh`。
