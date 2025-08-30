# YOLOv1 C++实现

这是一个YOLOv1（You Only Look Once）目标检测算法的C++实现版本。YOLOv1是一种高效的单阶段目标检测算法，能够实时检测多个对象。

## 项目结构

```
├── CMakeLists.txt      # CMake构建文件
├── include/            # 头文件目录
├── src/                # 源代码目录
└── examples/           # 示例代码目录
```

## 功能特点

- YOLOv1网络结构的C++实现
- 边界框预测和非极大值抑制(NMS)功能
- 损失函数计算
- 训练和推理功能

## 编译与安装

```bash
mkdir build && cd build
cmake ..
make
```

## 使用示例

### 目标检测

```cpp
// 创建YOLO检测器
YOLOParams params(7, 2, 20); // 7x7网格，每个网格2个边界框，20个类别
YOLODetector detector(params);

// 加载预训练模型
detector.loadWeights("yolov1_model.weights");

// 准备输入图像数据
std::vector<float> image = loadImage("image.jpg", 448, 448);

// 前向传播
std::vector<float> network_output = detector.forward(image);

// 解码网络输出为边界框
float confidence_threshold = 0.5f;
std::vector<BoundingBox> boxes = detector.decodeOutput(network_output, confidence_threshold);

// 应用非极大值抑制
float nms_threshold = 0.4f;
std::vector<BoundingBox> final_boxes = detector.applyNMS(boxes, nms_threshold);

// 处理检测结果
for (const auto& box : final_boxes) {
    std::cout << "Class: " << box.class_id 
              << ", Confidence: " << box.confidence 
              << ", Position: (" << box.x << ", " << box.y << ")"
              << ", Size: " << box.w << "x" << box.h << std::endl;
}
```

### 模型训练

#### 使用模拟数据训练

```cpp
// 创建YOLO检测器
YOLOParams params(7, 2, 20);
YOLODetector detector(params);

// 准备训练数据和标签
std::vector<std::vector<float>> training_data = loadTrainingData();
std::vector<std::vector<float>> labels = loadLabels();

// 设置训练参数
int epochs = 100;
float learning_rate = 0.001f;

// 开始训练
detector.train(training_data, labels, epochs, learning_rate);

// 保存训练好的模型
detector.saveWeights("yolov1_model.weights");
```

#### 使用真实数据集训练

1. 准备数据集（YOLO格式）
   - 图像文件（.jpg或.png）放在images目录
   - 标签文件（.txt）放在labels目录
   - 每个标签文件对应一个图像文件，格式为：`class_id x y w h`（每行一个对象）

2. 创建配置文件（例如`train_config.txt`）

```
# 数据集参数
dataset_type=YOLO
images_dir=/path/to/your/images
labels_dir=/path/to/your/labels

# 网络参数
grid_size=7
boxes_per_cell=2
num_classes=20

# 训练参数
epochs=100
batch_size=64
learning_rate=0.001
```

3. 运行训练脚本

```bash
./train_real_data.sh examples/train_config.txt
```

#### 使用CIFAR-10数据集训练

我们提供了专门的工具和脚本来训练CIFAR-10数据集：

1. 使用自动化脚本（推荐）

```bash
./scripts/prepare_cifar10.sh
```

该脚本会自动下载CIFAR-10数据集，将其转换为YOLO格式，并生成训练配置文件。

2. 手动步骤

```bash
# 编译项目
mkdir -p build && cd build
cmake ..
cmake --build .

# 转换数据集
./yolov1_cifar10_to_yolo /path/to/cifar10 /path/to/output

# 训练模型
./yolov1_train_real /path/to/output/cifar10_config.txt
```

详细信息请参阅[CIFAR-10训练指南](docs/cifar10_training_guide.md)。

详细的训练指南请参考：
- [基础训练指南](docs/training_guide.md)
- [真实数据集训练指南](docs/real_dataset_training.md)

## 依赖项

- C++17或更高版本
- CMake 3.10或更高版本
- （可选）OpenCV用于图像处理和可视化

## 参考

- [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)