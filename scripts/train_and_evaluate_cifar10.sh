#!/bin/bash

# 脚本：训练和评估CIFAR-10数据集上的YOLOv1模型

# 设置目录和参数
CIFAR10_DIR="./cifar10"
OUTPUT_DIR="./cifar10_yolo"
BUILD_DIR="./build"
MODEL_PATH="cifar10_model.weights"
VAL_IMAGES_DIR="${OUTPUT_DIR}/images/val"
VAL_LABELS_DIR="${OUTPUT_DIR}/labels/val"
CONFIG_PATH="${OUTPUT_DIR}/cifar10_config.txt"
IOU_THRESHOLD=0.5
CONF_THRESHOLD=0.5

# 检查数据集是否已准备
if [ ! -d "$OUTPUT_DIR/images/train" ] || [ ! -d "$OUTPUT_DIR/labels/train" ]; then
    echo "CIFAR-10数据集尚未转换为YOLO格式。先运行prepare_cifar10.sh脚本。"
    exit 1
fi

# 检查构建目录
if [ ! -d "$BUILD_DIR" ]; then
    echo "创建构建目录..."
    mkdir -p "$BUILD_DIR"
fi

# 编译项目
echo "编译项目..."
cd "$BUILD_DIR"
cmake ..
cmake --build .

# 训练模型
echo "\n开始训练YOLOv1模型..."
echo "配置文件: $CONFIG_PATH"
echo "模型将保存到: $MODEL_PATH"

# 使用绝对路径
ABSOLUTE_CONFIG_PATH="$(cd .. && pwd)/$CONFIG_PATH"
echo "使用绝对路径: $ABSOLUTE_CONFIG_PATH"
./yolov1_train_real "$ABSOLUTE_CONFIG_PATH"

# 检查训练是否成功
if [ ! -f "$MODEL_PATH" ]; then
    echo "训练失败，未生成模型文件。"
    exit 1
fi

# 评估模型
echo "\n开始评估YOLOv1模型..."
echo "验证集图像目录: $VAL_IMAGES_DIR"
echo "验证集标签目录: $VAL_LABELS_DIR"
echo "IoU阈值: $IOU_THRESHOLD"

# 使用绝对路径
ABSOLUTE_MODEL_PATH="$(cd .. && pwd)/$MODEL_PATH"
ABSOLUTE_VAL_IMAGES_DIR="$(cd .. && pwd)/$VAL_IMAGES_DIR"
ABSOLUTE_VAL_LABELS_DIR="$(cd .. && pwd)/$VAL_LABELS_DIR"
echo "使用绝对路径:"
echo "模型: $ABSOLUTE_MODEL_PATH"
echo "验证集图像: $ABSOLUTE_VAL_IMAGES_DIR"
echo "验证集标签: $ABSOLUTE_VAL_LABELS_DIR"
echo "置信度阈值: $CONF_THRESHOLD"
./yolov1_cifar10_evaluate "$ABSOLUTE_MODEL_PATH" "$ABSOLUTE_VAL_IMAGES_DIR" "$ABSOLUTE_VAL_LABELS_DIR" "$IOU_THRESHOLD" "$CONF_THRESHOLD"

echo "\n训练和评估完成！"