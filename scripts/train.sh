#!/bin/bash

# YOLOv1训练脚本

# 确保build目录存在
mkdir -p build
cd build

# 编译项目
cmake ..
cmake --build .

# 运行训练示例
./yolov1_train

echo "训练完成！模型已保存到 yolov1_model.weights"