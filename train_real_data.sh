#!/bin/bash

# 检查配置文件参数
if [ $# -lt 1 ]; then
    echo "Usage: $0 <config_file>"
    echo "Example: $0 examples/train_config.txt"
    exit 1
fi

CONFIG_FILE=$1

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found!"
    exit 1
fi

# 创建build目录（如果不存在）
mkdir -p build
cd build

# 编译项目
echo "Compiling project..."
cmake ..
cmake --build .

if [ $? -ne 0 ]; then
    echo "Error: Compilation failed!"
    exit 1
fi

# 运行训练程序
echo "\nStarting training with config file: $CONFIG_FILE"
./yolov1_train_real "../$CONFIG_FILE"