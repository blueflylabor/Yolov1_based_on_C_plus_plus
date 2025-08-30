# YOLOv1 训练指南

本文档提供了如何使用YOLOv1 C++实现进行模型训练的详细指南。

## 目录

1. [准备训练数据](#准备训练数据)
2. [配置训练参数](#配置训练参数)
3. [训练模型](#训练模型)
4. [保存和加载模型](#保存和加载模型)
5. [评估模型性能](#评估模型性能)
6. [常见问题](#常见问题)

## 准备训练数据

### 数据格式

YOLOv1需要的训练数据包括两部分：

1. **输入图像**：通常是448x448像素的RGB图像，归一化到[0,1]范围。
2. **标签数据**：对应于网络输出格式的标签，大小为S×S×(B×5+C)，其中：
   - S是网格大小（默认为7）
   - B是每个网格单元预测的边界框数量（默认为2）
   - C是类别数量（默认为20）

### 标签编码

对于每个训练样本，标签数据的编码方式如下：

1. 将图像划分为S×S网格。
2. 对于每个对象，确定其中心点所在的网格单元(i,j)。
3. 在该网格单元的标签中设置：
   - 边界框坐标(x,y,w,h)，其中x和y是相对于整个图像的归一化坐标(0-1)，w和h是相对于整个图像的归一化宽度和高度。
   - 置信度值设为1（表示有对象）。
   - 对应类别的概率设为1，其他类别为0。

### 示例代码

以下是创建训练数据的示例代码（简化版）：

```cpp
// 创建模拟的训练数据
std::vector<std::vector<float>> createTrainingData(int num_samples, int width, int height, int channels) {
    std::vector<std::vector<float>> training_data;
    
    // 创建num_samples个训练样本
    for (int i = 0; i < num_samples; i++) {
        std::vector<float> image(width * height * channels);
        // 填充图像数据（这里简化为随机值）
        for (size_t j = 0; j < image.size(); j++) {
            image[j] = static_cast<float>(rand()) / RAND_MAX;
        }
        training_data.push_back(image);
    }
    
    return training_data;
}

// 创建标签数据
std::vector<std::vector<float>> createLabels(int num_samples, int S, int B, int num_classes) {
    std::vector<std::vector<float>> labels;
    int output_size = S * S * (B * 5 + num_classes);
    
    for (int i = 0; i < num_samples; i++) {
        std::vector<float> label(output_size, 0.0f);
        
        // 为每个样本添加对象（这里简化为随机位置）
        int num_objects = 1 + rand() % 3; // 每个图像1-3个对象
        
        for (int obj = 0; obj < num_objects; obj++) {
            // 随机选择网格位置
            int grid_i = rand() % S;
            int grid_j = rand() % S;
            int grid_index = grid_i * S + grid_j;
            int offset = grid_index * (B * 5 + num_classes);
            
            // 设置边界框参数
            label[offset + 0] = (grid_j + 0.5f) / S; // x中心点
            label[offset + 1] = (grid_i + 0.5f) / S; // y中心点
            label[offset + 2] = 0.2f; // 宽度
            label[offset + 3] = 0.2f; // 高度
            label[offset + 4] = 1.0f; // 置信度
            
            // 设置类别（随机选择一个类别）
            int class_id = rand() % num_classes;
            label[offset + B * 5 + class_id] = 1.0f;
        }
        
        labels.push_back(label);
    }
    
    return labels;
}
```

## 配置训练参数

### YOLOParams结构体

训练前需要配置YOLOParams结构体，设置网络参数：

```cpp
YOLOParams params;
params.S = 7;              // 网格大小
params.B = 2;              // 每个网格单元预测的边界框数量
params.num_classes = 20;   // 类别数量
params.lambda_coord = 5.0; // 坐标损失权重
params.lambda_noobj = 0.5; // 无对象损失权重
```

### 训练超参数

主要的训练超参数包括：

- **学习率**：通常从0.001开始，可以使用学习率衰减策略。
- **批量大小**：根据可用内存设置，通常为64或128。
- **训练轮数**：通常为100-200轮，根据数据集大小和收敛情况调整。
- **权重衰减**：通常设为0.0005，用于正则化。
- **动量**：通常设为0.9，用于优化器。

## 训练模型

### 基本训练流程

```cpp
// 创建YOLO检测器
YOLOParams params(7, 2, 20);
YOLODetector detector(params);

// 准备训练数据和标签
std::vector<std::vector<float>> training_data = createTrainingData(num_samples, 448, 448, 3);
std::vector<std::vector<float>> labels = createLabels(num_samples, params.S, params.B, params.num_classes);

// 设置训练参数
int epochs = 100;
float learning_rate = 0.001f;

// 开始训练
detector.train(training_data, labels, epochs, learning_rate);

// 保存模型
detector.saveWeights("yolov1_model.weights");
```

### 高级训练技巧

1. **数据增强**：
   - 随机缩放和裁剪
   - 随机调整亮度、对比度、饱和度
   - 随机水平翻转（注意需要调整边界框坐标）

2. **学习率调整**：
   - 开始时使用较小的学习率（如0.0001）预热几个epoch
   - 然后增加到0.001
   - 当损失平稳时，将学习率降低10倍

3. **正则化**：
   - 使用权重衰减（L2正则化）
   - 使用Dropout（通常在全连接层）

## 保存和加载模型

### 保存模型

```cpp
bool success = detector.saveWeights("yolov1_model.weights");
if (success) {
    std::cout << "Model saved successfully!" << std::endl;
} else {
    std::cerr << "Failed to save model!" << std::endl;
}
```

### 加载模型

```cpp
YOLOParams params(7, 2, 20);
YOLODetector detector(params);

bool success = detector.loadWeights("yolov1_model.weights");
if (success) {
    std::cout << "Model loaded successfully!" << std::endl;
} else {
    std::cerr << "Failed to load model!" << std::endl;
}
```

## 评估模型性能

### 计算mAP（平均精度均值）

评估目标检测模型性能的标准指标是mAP（mean Average Precision）。计算步骤：

1. 对每个类别，计算不同置信度阈值下的精度和召回率
2. 绘制PR曲线（精度-召回率曲线）
3. 计算PR曲线下的面积（AP）
4. 对所有类别的AP取平均值得到mAP

### 示例评估代码框架

```cpp
// 评估函数框架
void evaluateModel(YOLODetector& detector, const std::vector<std::vector<float>>& test_images,
                  const std::vector<std::vector<BoundingBox>>& ground_truth_boxes) {
    // 存储每个类别的检测结果
    std::vector<std::vector<Detection>> all_detections(num_classes);
    
    // 对每个测试图像进行预测
    for (size_t i = 0; i < test_images.size(); i++) {
        // 前向传播
        std::vector<float> output = detector.forward(test_images[i]);
        
        // 解码输出为边界框
        std::vector<BoundingBox> boxes = detector.decodeOutput(output, 0.01f); // 使用低置信度阈值
        
        // 应用NMS
        std::vector<BoundingBox> final_boxes = detector.applyNMS(boxes, 0.45f);
        
        // 存储检测结果
        for (const auto& box : final_boxes) {
            all_detections[box.class_id].push_back({box, i});
        }
    }
    
    // 计算每个类别的AP
    std::vector<float> APs(num_classes, 0.0f);
    for (int c = 0; c < num_classes; c++) {
        APs[c] = calculateAP(all_detections[c], ground_truth_boxes, c);
        std::cout << "AP for class " << c << ": " << APs[c] << std::endl;
    }
    
    // 计算mAP
    float mAP = 0.0f;
    for (float ap : APs) {
        mAP += ap;
    }
    mAP /= num_classes;
    
    std::cout << "mAP: " << mAP << std::endl;
}
```

## 常见问题

### 1. 损失不下降

可能的原因和解决方案：

- **学习率过高**：尝试降低学习率
- **数据预处理问题**：检查图像归一化和标签编码
- **网络初始化问题**：尝试不同的权重初始化方法
- **梯度爆炸/消失**：检查激活函数和权重初始化

### 2. 过拟合

可能的解决方案：

- **增加训练数据**：使用数据增强
- **添加正则化**：增加权重衰减或Dropout
- **简化模型**：减少网络参数

### 3. 检测精度低

可能的解决方案：

- **调整损失函数权重**：增加lambda_coord或减小lambda_noobj
- **调整NMS参数**：尝试不同的IOU阈值
- **增加训练轮数**：让模型充分收敛
- **调整置信度阈值**：在推理时尝试不同的置信度阈值

### 4. 训练速度慢

可能的解决方案：

- **减小批量大小**：如果内存不足
- **使用GPU加速**：如果可用
- **优化数据加载**：使用多线程数据加载
- **减少数据增强复杂度**：在初期训练阶段

## 结论

训练YOLOv1模型需要合理的数据准备、参数配置和训练策略。通过本指南提供的方法和技巧，您应该能够成功训练出自己的YOLOv1目标检测模型。如果遇到特定问题，请参考常见问题部分或查阅更详细的文档。