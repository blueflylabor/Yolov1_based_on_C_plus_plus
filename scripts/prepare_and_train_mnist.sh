#!/bin/bash

# 设置工作目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
MNIST_DIR="$BUILD_DIR/mnist_yolo"
MNIST_DATA_DIR="$BUILD_DIR/mnist_data"

# 创建必要的目录
mkdir -p "$MNIST_DATA_DIR"
mkdir -p "$MNIST_DIR"
mkdir -p "$MNIST_DIR/images/train"
mkdir -p "$MNIST_DIR/images/test"
mkdir -p "$MNIST_DIR/labels/train"
mkdir -p "$MNIST_DIR/labels/test"

# 复制配置文件
cp "$PROJECT_DIR/mnist_config.txt" "$MNIST_DIR/"

# 设置MNIST数据集URL
TRAIN_IMAGES_URL="http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
TRAIN_LABELS_URL="http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
TEST_IMAGES_URL="http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
TEST_LABELS_URL="http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"

# 下载并解压MNIST数据集
download_and_extract() {
    local url=$1
    local filename=$(basename "$url")
    local extracted_file=${filename%.gz}
    
    if [ ! -f "$MNIST_DATA_DIR/$extracted_file" ]; then
        echo "下载 $filename..."
        curl -L "$url" -o "$MNIST_DATA_DIR/$filename"
        
        echo "解压 $filename..."
        gunzip -f "$MNIST_DATA_DIR/$filename"
    else
        echo "$extracted_file 已存在，跳过下载"
    fi
}

echo "=== 下载MNIST数据集 ==="
download_and_extract "$TRAIN_IMAGES_URL"
download_and_extract "$TRAIN_LABELS_URL"
download_and_extract "$TEST_IMAGES_URL"
download_and_extract "$TEST_LABELS_URL"

# 编译项目
echo "=== 编译项目 ==="
cd "$PROJECT_DIR"

if [ ! -d "$BUILD_DIR" ]; then
    mkdir -p "$BUILD_DIR"
fi

cd "$BUILD_DIR"
cmake ..
cmake --build .

# 转换MNIST数据集为YOLO格式
echo "=== 转换MNIST数据集为YOLO格式 ==="
"$BUILD_DIR/yolov1_mnist_to_yolo" \
    "$MNIST_DATA_DIR/train-images-idx3-ubyte" \
    "$MNIST_DATA_DIR/train-labels-idx1-ubyte" \
    "$MNIST_DIR"

# 获取配置文件的绝对路径
ABSOLUTE_CONFIG_PATH="$MNIST_DIR/mnist_config.txt"

# 训练模型
echo "=== 开始训练MNIST模型 ==="
cd "$BUILD_DIR"
./yolov1_train_real "$ABSOLUTE_CONFIG_PATH"

echo "=== MNIST训练完成 ==="