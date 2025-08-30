# CIFAR-10 数据集训练指南

本文档提供了如何使用YOLOv1 C++实现对CIFAR-10数据集进行训练的详细指南。

## 目录

1. [CIFAR-10数据集简介](#cifar-10数据集简介)
2. [准备数据集](#准备数据集)
3. [转换为YOLO格式](#转换为yolo格式)
4. [配置训练参数](#配置训练参数)
5. [训练模型](#训练模型)
6. [评估模型性能](#评估模型性能)
7. [常见问题](#常见问题)

## CIFAR-10数据集简介

CIFAR-10是一个广泛使用的计算机视觉数据集，由加拿大高等研究院(CIFAR)收集。它包含10个类别的60,000张32x32彩色图像，每个类别6,000张图像。数据集分为50,000张训练图像和10,000张测试图像。

### 类别

CIFAR-10包含以下10个类别：
1. 飞机(airplane)
2. 汽车(automobile)
3. 鸟(bird)
4. 猫(cat)
5. 鹿(deer)
6. 狗(dog)
7. 青蛙(frog)
8. 马(horse)
9. 船(ship)
10. 卡车(truck)

### 数据格式

CIFAR-10原始数据集以二进制格式提供，每个图像由一个标签字节和3072个像素字节组成（32×32×3）。为了使用YOLOv1进行训练，我们需要将其转换为YOLO格式。

## 准备数据集

我们提供了一个脚本来自动下载和准备CIFAR-10数据集。按照以下步骤操作：

1. 打开终端并导航到项目根目录
2. 运行准备脚本：

```bash
./scripts/prepare_cifar10.sh
```

该脚本将：
- 下载CIFAR-10数据集
- 解压数据集
- 编译项目
- 将数据集转换为YOLO格式
- 创建训练配置文件

## 转换为YOLO格式

如果您想手动转换数据集，可以按照以下步骤操作：

1. 下载CIFAR-10数据集：

```bash
mkdir -p cifar10
curl -L "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz" -o "cifar10/cifar-10-binary.tar.gz"
tar -xzf "cifar10/cifar-10-binary.tar.gz" -C "cifar10"
```

2. 编译项目：

```bash
mkdir -p build && cd build
cmake ..
cmake --build .
```

3. 运行转换工具：

```bash
./yolov1_cifar10_to_yolo /path/to/cifar10/cifar-10-batches-bin /path/to/output/directory
```

转换工具将：
- 读取CIFAR-10二进制文件
- 将图像保存为PNG格式
- 创建YOLO格式的标签文件
- 生成训练配置文件

## 配置训练参数

转换工具会自动生成一个配置文件`cifar10_config.txt`，其中包含以下参数：

```
# CIFAR-10训练配置文件
dataset_type=YOLO
images_dir=/path/to/images/train
labels_dir=/path/to/labels/train

# 图像参数
image_width=448
image_height=448

# 网络参数
grid_size=7
boxes_per_cell=2
num_classes=10  # CIFAR-10有10个类别
lambda_coord=5.0
lambda_noobj=0.5

# 训练参数
epochs=100
batch_size=64
learning_rate=0.001
weight_decay=0.0005
momentum=0.9

# 模型保存参数
model_save_path=cifar10_model.weights
```

您可以根据需要调整这些参数。特别是，您可能需要调整以下参数：

- **epochs**：训练轮数，可以根据收敛情况增加或减少
- **batch_size**：批量大小，可以根据可用内存调整
- **learning_rate**：学习率，可以尝试不同的值以获得更好的性能
- **image_width/image_height**：输入图像尺寸，注意CIFAR-10原始图像为32x32，需要调整大小

## 训练模型

准备好数据集和配置文件后，您可以开始训练模型：

```bash
cd build
./yolov1_train_real /path/to/cifar10_config.txt
```

训练过程将显示每个epoch的损失值，并在训练结束后保存模型权重。

## 评估模型性能

训练完成后，您可以使用我们提供的评估工具来评估模型在CIFAR-10测试集上的性能。

### 使用评估脚本

我们提供了一个脚本来自动训练和评估模型：

```bash
./scripts/train_and_evaluate_cifar10.sh
```

该脚本将：
- 编译项目
- 训练YOLOv1模型
- 评估模型性能
- 输出评估指标

### 手动评估

如果您想手动评估模型，可以按照以下步骤操作：

1. 编译项目（如果尚未编译）：

```bash
mkdir -p build && cd build
cmake ..
cmake --build .
```

2. 运行评估工具：

```bash
./yolov1_cifar10_evaluate /path/to/model.weights /path/to/images/val /path/to/labels/val [iou_threshold] [conf_threshold]
```

参数说明：
- `/path/to/model.weights`：训练好的模型权重文件路径
- `/path/to/images/val`：验证集图像目录
- `/path/to/labels/val`：验证集标签目录
- `iou_threshold`：IoU阈值，默认为0.5
- `conf_threshold`：置信度阈值，默认为0.5

### 评估指标

评估工具将输出以下指标：

- **精确率(Precision)**：正确预测的比例
- **召回率(Recall)**：找到的真实目标的比例
- **F1分数**：精确率和召回率的调和平均值
- **mAP(平均精度均值)**：所有类别的平均精度
- **总体准确率**：正确预测的总比例
- **类别准确率**：每个类别的准确率

这些指标可以帮助您了解模型的性能，并指导进一步的改进。


## 常见问题

### Q: 转换数据集时出现错误

**A**: 确保您已正确下载并解压CIFAR-10数据集。检查目录结构是否正确，以及是否有足够的磁盘空间。

### Q: 训练过程中内存不足

**A**: 尝试减小批量大小(batch_size)。在配置文件中将`batch_size`参数设置为较小的值，如16或32。

### Q: 模型性能不佳

**A**: 尝试以下方法改进模型性能：
- 增加训练轮数(epochs)
- 调整学习率(learning_rate)
- 修改网格大小(grid_size)
- 增加数据增强

### Q: 评估工具无法找到图像文件

**A**: 确保图像文件扩展名正确(.png)，并且图像和标签目录路径正确。

### 1. 图像尺寸问题

CIFAR-10图像尺寸为32x32，而YOLOv1通常使用448x448的输入。解决方法：

- 在转换过程中将图像调整为448x448
- 或修改网络结构以适应32x32的输入

### 2. 训练速度慢

如果训练速度太慢，可以尝试：

- 减小批量大小
- 使用更少的训练样本进行初步测试
- 减少网络复杂度

### 3. 过拟合问题

CIFAR-10数据集相对较小，可能会出现过拟合。解决方法：

- 增加数据增强
- 增加权重衰减
- 使用早停法

### 4. 内存不足

如果遇到内存不足问题，可以尝试：

- 减小批量大小
- 减少训练样本数量
- 使用内存效率更高的实现