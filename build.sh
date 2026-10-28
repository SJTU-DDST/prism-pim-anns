#!/bin/bash

# 1. 设置 CMake 路径
CMAKE_BIN_DIR="$HOME/cmake-3.31.6/bin"

# 2. 配置项目 (-S 源目录 -B 构建目录)
# 只有当 CMakeLists.txt 发生变化时才需要重新运行这一步，但多跑一次也无妨
"$CMAKE_BIN_DIR/cmake" -S . -B build
#     -DCMAKE_BUILD_TYPE=Release

# 3. 编译 (-j 自动利用所有核)
# 这一步生成可执行文件在 build/ 目录下
make -C build -j

# 4. 运行
# ./build/host_cpp
