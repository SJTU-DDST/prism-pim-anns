# Prism: Reducing Arithmetic for High-Recall Approximate Nearest Neighbor Search on Processing-in-Memory

[简体中文](README.md) | **English**

Prism is a PIM-based ANNS system co-optimizing vector pruning, distance evaluation, and host-PIM orchestration.
It employs a proximity-aware vector pruner to leverage high intra-PIM bandwidth and dual-cluster affiliations to filter out distant vectors.
Prism then performs sensitivity-ordered distance computation, prioritizing high-impact dimension segments and terminating distance computation for candidate vectors early once exclusion criteria are met.
A stall-free host-PIM pipeline overlaps query preparation, PIM execution, and global ranking to sustain high throughput.
Experiments on UPMEM PIM show that Prism achieves 2.7–19.8 times the throughput of state-of-the-art systems.

## Repository Layout

The default dataset path is `/mnt/anns-dataset`. `./dpu` contains the DPU code, `./host` contains the host code, and `./share` contains variables, macros, and other definitions shared by the host and DPUs. `./plot` contains Python scripts for plotting.

## Build Dependencies

The project uses the system UPMEM SDK installed under `/usr`, including the DPU compiler, host headers, and `libdpu`. Building also requires CMake 3.15 or later, a compiler with C++20 support, OpenMP, and Faiss.

## Building and Running the Code

### Installing Faiss

1. The C++ code depends on Faiss for clustering. Build and install Faiss from source. First, install the required build dependencies:

   ```bash
   sudo apt-get install libgflags-dev libblas-dev liblapack-dev swig
   ```

2. Download or clone the Faiss source code from GitHub and enter its directory after extraction. If you downloaded a ZIP archive, rename `faiss-main` to `faiss`.

   Create the build directory:

   ```bash
   mkdir build
   ```

   Configure the build. Our UPMEM host uses an Intel Xeon Silver 4216 CPU @ 2.10GHz, which supports Intel BLAS but does not support `avx512_spr`:

   ```bash
   cmake -B build . \
       -DFAISS_ENABLE_GPU=OFF \
       -DBUILD_TESTING=OFF \
       -DFAISS_OPT_LEVEL=avx512 \
       -DCMAKE_BUILD_TYPE=Release \
       -DBLA_VENDOR=Intel10_64_dyn
   ```

   Build:

   ```bash
   make -C build -j
   ```

   Install:

   ```bash
   sudo make -C build install
   ```

   When Intel MKL is enabled, the compiled library may fail to locate the MKL libraries. For example, `ldd` may report `libmkl_rt.so.2 => not found`. Add the MKL installation directory to your environment:

   ```bash
   echo 'export LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/2025.2/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc
   ```

3. Copy the `impl` directory from the `faiss` subdirectory of the extracted source tree to `/usr/local/include/faiss/impl`; otherwise, some headers may not be found:

   ```bash
   sudo cp -r ~/faiss/faiss/impl /usr/local/include/faiss/impl
   ```

### Building and Running Prism

1. Configure and build from the project root:

   ```bash
   cmake -S . -B build -DQUERY_MODE=3 -DNR_TASKLETS=16
   cmake --build build -j
   ```

2. Run from the project root, since some files are accessed through relative paths:

   ```bash
   ./build/host_app
   ```

3. On our UPMEM server, the CPU defaults to the `powersave` governor after each reboot. Check the current governor with `cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor` and switch to performance mode with `echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`.

### Parameter Locations

1. Parameters shared by the host and DPUs, such as the dataset and dataset size, are mainly defined in `./share/global.h`. DPU parameters, such as optimization methods, are mainly defined in `./dpu/dpu_global.c`. Host parameters, such as the query method, are mainly defined in `./host/config.cc`. Batch parameters are defined in `./host/utils.hh`.
2. Set the number of DPU threads and the query mode through CMake, for example: `cmake -S . -B build -DNR_TASKLETS=16 -DQUERY_MODE=3`. `QUERY_MODE` supports 0 (Sequential), 1 (Pipeline), 2 (Callback), and 3 (Async Pipeline), with 3 as the default. Reconfigure and rebuild to apply changes.
3. Run experiments with the Bash scripts. `run_baseline.sh`, `run_cpu_baseline.sh`, and `run_ours.sh` launch the program once for each combination of dataset, query mode, batch settings, and other configuration options. For the 100M datasets, nprobe values are defined in `get_qc_num_sweep()` in `host/utils.cc` and selected according to `DATA_NAME`, `TOP_K`, and the query method. The CPU baseline uses a separate set of values.
4. Command-line arguments are managed by cxxopts. Run `./build/host_app --help` to view options for data paths, query methods, query counts, and other settings.

### Enabling Prism Components and the Baseline

1. To run the DPU baseline, set `MIN_REPLICA` and `MAX_REPLICA` in `./share/global.h` to 1, and set `L2_DIST_METHOD` in `./dpu/dpu_global.h` to 0. Alternatively, run `./run_dpu_baseline.sh` to use this configuration directly.
2. To enable the Host-PIM Stall-Free Execution Pipeline, compile with `-DQUERY_MODE=3`.
3. To enable Intersection-Based Vector Pruning, set `MIN_REPLICA` and `MAX_REPLICA` in `./share/global.h` to 2. `IS_CROSS_QUERY`, which controls intersection-based querying, is automatically set to `true`.
4. To enable Sensitivity-Ordered Distance Computation, set `L2_DIST_METHOD` in `./dpu/dpu_global.h` to 1.
5. Combining components 2–4 gives the Prism configuration. Run `./run_ours.sh` to use this configuration directly.
