#!/bin/bash

# 设置工作目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"

# 创建测试数据目录
TEST_DIR="$BUILD_DIR/test_data"
mkdir -p "$TEST_DIR/images/train"
mkdir -p "$TEST_DIR/labels/train"

# 创建简单的测试图像和标签
echo "创建测试数据..."

# 确保目录存在后再获取绝对路径
ABS_IMAGES_DIR="$BUILD_DIR/test_data/images/train"
ABS_LABELS_DIR="$BUILD_DIR/test_data/labels/train"

# 创建一个简单的测试配置文件
cat > "$TEST_DIR/test_config.txt" << EOF
# 测试配置文件

# 数据集配置
dataset_type = test
images_dir = $ABS_IMAGES_DIR
labels_dir = $ABS_LABELS_DIR
image_width = 28
image_height = 28

# 网络参数
grid_size = 7
boxes_per_cell = 2
num_classes = 10
coord_scale = 5.0
noobj_scale = 0.5

# 训练参数
epochs = 3
batch_size = 2
learning_rate = 0.001
weight_decay = 0.0005
momentum = 0.9

# 模型保存
model_save_path = ./models/yolov1_test.weights
EOF

# 创建一个简单的测试图像文件 - 使用base64编码的1x1像素PNG图像
BASE64_PNG="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVQI12P4//8/AAX+Av7czFnnAAAAAElFTkSuQmCC"

for i in {1..5}; do
    # 解码base64并保存为PNG文件
    echo "$BASE64_PNG" | base64 -d > "$TEST_DIR/images/train/image_$i.png"
    
    # 创建对应的YOLO格式标签文件
    echo "0 0.5 0.5 0.5 0.5" > "$TEST_DIR/labels/train/image_$i.txt"
done

echo "测试数据创建完成"

# 运行训练程序
echo "开始测试损失计算..."
cd "$BUILD_DIR"
./yolov1_train_real "$TEST_DIR/test_config.txt"

echo "测试完成"