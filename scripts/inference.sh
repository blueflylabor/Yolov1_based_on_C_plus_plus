#!/bin/bash

# YOLOv1推理脚本

# 确保build目录存在
mkdir -p build
cd build

# 编译项目
cmake ..
cmake --build .

# 运行推理示例
./yolov1_demo

echo "推理完成！"