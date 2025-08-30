#!/bin/bash

# 脚本：准备CIFAR-10数据集并转换为YOLO格式

# 设置目录
CIFAR10_DIR="./cifar10"
OUTPUT_DIR="./cifar10_yolo"
BUILD_DIR="./build"

# 创建目录
mkdir -p "$CIFAR10_DIR"
mkdir -p "$OUTPUT_DIR"

# 下载CIFAR-10数据集
echo "下载CIFAR-10数据集..."
if [ ! -f "$CIFAR10_DIR/cifar-10-binary.tar.gz" ]; then
    curl -L "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz" -o "$CIFAR10_DIR/cifar-10-binary.tar.gz"
else
    echo "CIFAR-10数据集已下载"
fi

# 解压数据集
echo "解压CIFAR-10数据集..."
if [ ! -d "$CIFAR10_DIR/cifar-10-batches-bin" ]; then
    tar -xzf "$CIFAR10_DIR/cifar-10-binary.tar.gz" -C "$CIFAR10_DIR"
else
    echo "CIFAR-10数据集已解压"
fi

# 编译项目
echo "编译项目..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake ..
cmake --build .

# 返回项目根目录
cd ..

# 转换数据集
echo "转换CIFAR-10数据集为YOLO格式..."
# 使用绝对路径
PROJECT_DIR="$(pwd)"
CIFAR10_ABSOLUTE_PATH="$PROJECT_DIR/cifar10/cifar-10-batches-bin"
OUTPUT_ABSOLUTE_PATH="$PROJECT_DIR/cifar10_yolo"
echo "使用绝对路径: $CIFAR10_ABSOLUTE_PATH"
$BUILD_DIR/yolov1_cifar10_to_yolo "$CIFAR10_ABSOLUTE_PATH" "$OUTPUT_ABSOLUTE_PATH"

# 修改文件扩展名
echo "更新文件扩展名..."
for img in "$OUTPUT_DIR/images/train"/*.ppm "$OUTPUT_DIR/images/val"/*.ppm; do
    if [ -f "$img" ]; then
        mv "$img" "${img%.ppm}.png"
    fi
done

# 检查转换是否成功
if [ $? -eq 0 ]; then
    echo "\n转换成功！现在可以使用以下命令训练YOLOv1模型："
    echo "./yolov1_train_real $OUTPUT_DIR/cifar10_config.txt"
else
    echo "\n转换失败，请检查错误信息"
fi