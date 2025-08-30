# 使用YOLOv1 C++实现对真实数据集进行训练和检测

本文档提供了如何使用YOLOv1 C++实现对真实数据集进行训练和检测的详细指南。

## 目录

1. [支持的数据集格式](#支持的数据集格式)
2. [数据集准备](#数据集准备)
3. [数据预处理](#数据预处理)
4. [修改代码支持真实数据集](#修改代码支持真实数据集)
5. [训练配置](#训练配置)
6. [训练过程](#训练过程)
7. [评估模型](#评估模型)
8. [实际检测](#实际检测)
9. [常见问题](#常见问题)

## 支持的数据集格式

本实现可以支持以下常见的目标检测数据集格式：

1. **PASCAL VOC格式**：XML标注文件，每个图像对应一个XML文件
2. **COCO格式**：JSON标注文件，包含所有图像的标注信息
3. **YOLO格式**：TXT标注文件，每个图像对应一个TXT文件

## 数据集准备

### 1. 获取公开数据集

您可以使用以下公开数据集进行训练：

- **PASCAL VOC**：[下载链接](http://host.robots.ox.ac.uk/pascal/VOC/)
- **COCO**：[下载链接](https://cocodataset.org/#download)
- **Open Images**：[下载链接](https://storage.googleapis.com/openimages/web/index.html)

### 2. 准备自定义数据集

如果您想使用自定义数据集，需要按照以下步骤准备：

1. **收集图像**：收集包含目标对象的图像
2. **标注图像**：使用标注工具（如LabelImg、CVAT等）标注图像中的对象
3. **组织数据**：将图像和标注文件组织成支持的格式之一

#### 推荐的标注工具

- **LabelImg**：[GitHub链接](https://github.com/tzutalin/labelImg)
- **CVAT**：[GitHub链接](https://github.com/opencv/cvat)
- **VoTT**：[GitHub链接](https://github.com/microsoft/VoTT)

## 数据预处理

### 1. 数据转换

无论您使用哪种格式的数据集，都需要将其转换为YOLOv1所需的格式。以下是将不同格式转换为YOLOv1格式的示例代码：

```cpp
// 添加到项目中的新文件：data_loader.h
#ifndef YOLOV1_DATA_LOADER_H
#define YOLOV1_DATA_LOADER_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <filesystem>
#include "yolo.h"

namespace fs = std::filesystem;

class DataLoader {
public:
    // 加载YOLO格式数据集
    static std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> 
    loadYOLODataset(const std::string& images_dir, const std::string& labels_dir, 
                    int image_width, int image_height, int channels,
                    int S, int B, int num_classes) {
        std::vector<std::vector<float>> images;
        std::vector<std::vector<float>> labels;
        
        // 遍历图像目录
        for (const auto& entry : fs::directory_iterator(images_dir)) {
            if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
                std::string image_path = entry.path().string();
                std::string filename = entry.path().stem().string();
                std::string label_path = labels_dir + "/" + filename + ".txt";
                
                // 检查对应的标签文件是否存在
                if (fs::exists(label_path)) {
                    // 加载图像（这里需要使用OpenCV或其他图像处理库）
                    std::vector<float> image = loadAndPreprocessImage(image_path, image_width, image_height);
                    
                    // 加载标签并转换为YOLOv1格式
                    std::vector<float> label = convertYOLOLabelToYOLOv1Format(
                        loadYOLOLabel(label_path), S, B, num_classes);
                    
                    images.push_back(image);
                    labels.push_back(label);
                }
            }
        }
        
        return {images, labels};
    }
    
    // 加载VOC格式数据集
    static std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> 
    loadVOCDataset(const std::string& images_dir, const std::string& annotations_dir, 
                  int image_width, int image_height, int channels,
                  int S, int B, int num_classes, const std::vector<std::string>& class_names) {
        // 类似于loadYOLODataset，但解析VOC XML格式
        // 此处省略实现...
    }
    
    // 加载COCO格式数据集
    static std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> 
    loadCOCODataset(const std::string& images_dir, const std::string& annotation_file, 
                   int image_width, int image_height, int channels,
                   int S, int B, int num_classes) {
        // 此处省略实现...
    }
    
private:
    // 加载并预处理图像
    static std::vector<float> loadAndPreprocessImage(const std::string& image_path, 
                                                    int width, int height) {
        // 使用OpenCV加载图像并调整大小
        // 此处省略实现...
        
        // 返回归一化的图像数据
        std::vector<float> image_data(width * height * 3, 0.0f);
        return image_data;
    }
    
    // 加载YOLO格式标签
    static std::vector<BoundingBox> loadYOLOLabel(const std::string& label_path) {
        std::vector<BoundingBox> boxes;
        std::ifstream file(label_path);
        
        if (file.is_open()) {
            std::string line;
            while (std::getline(file, line)) {
                std::istringstream iss(line);
                int class_id;
                float x, y, w, h;
                
                if (!(iss >> class_id >> x >> y >> w >> h)) {
                    continue; // 格式错误，跳过
                }
                
                BoundingBox box;
                box.class_id = class_id;
                box.x = x;
                box.y = y;
                box.w = w;
                box.h = h;
                box.confidence = 1.0f; // 真实框的置信度为1
                
                boxes.push_back(box);
            }
            file.close();
        }
        
        return boxes;
    }
    
    // 将YOLO格式标签转换为YOLOv1格式
    static std::vector<float> convertYOLOLabelToYOLOv1Format(
        const std::vector<BoundingBox>& boxes, int S, int B, int num_classes) {
        
        int output_size = S * S * (B * 5 + num_classes);
        std::vector<float> label(output_size, 0.0f);
        
        // 对每个边界框
        for (const auto& box : boxes) {
            // 确定中心点所在的网格单元
            int grid_i = static_cast<int>(box.y * S);
            int grid_j = static_cast<int>(box.x * S);
            
            // 确保网格索引在有效范围内
            grid_i = std::min(std::max(0, grid_i), S - 1);
            grid_j = std::min(std::max(0, grid_j), S - 1);
            
            // 计算中心点相对于网格单元的偏移
            float x_offset = box.x * S - grid_j;
            float y_offset = box.y * S - grid_i;
            
            // 计算标签数组中的索引
            int grid_index = grid_i * S + grid_j;
            int offset = grid_index * (B * 5 + num_classes);
            
            // 设置边界框参数（对于第一个预测框）
            label[offset + 0] = box.x; // x中心点（相对于整个图像）
            label[offset + 1] = box.y; // y中心点（相对于整个图像）
            label[offset + 2] = box.w; // 宽度（相对于整个图像）
            label[offset + 3] = box.h; // 高度（相对于整个图像）
            label[offset + 4] = 1.0f;  // 置信度（有对象）
            
            // 设置类别
            label[offset + B * 5 + box.class_id] = 1.0f;
        }
        
        return label;
    }
};

#endif // YOLOV1_DATA_LOADER_H
```

### 2. 数据增强

为了提高模型的泛化能力，建议对训练数据进行增强。以下是一个简单的数据增强实现：

```cpp
// 添加到data_loader.h中
class DataAugmentation {
public:
    // 随机水平翻转
    static void randomHorizontalFlip(std::vector<float>& image, std::vector<float>& label,
                                    int width, int height, int channels,
                                    int S, int B, int num_classes, float probability = 0.5) {
        if (static_cast<float>(rand()) / RAND_MAX < probability) {
            // 翻转图像
            std::vector<float> flipped_image(image.size());
            for (int c = 0; c < channels; c++) {
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        int src_idx = c * width * height + h * width + w;
                        int dst_idx = c * width * height + h * width + (width - 1 - w);
                        flipped_image[dst_idx] = image[src_idx];
                    }
                }
            }
            image = flipped_image;
            
            // 翻转标签中的边界框
            for (int i = 0; i < S; i++) {
                for (int j = 0; j < S; j++) {
                    int grid_index = i * S + j;
                    int offset = grid_index * (B * 5 + num_classes);
                    
                    // 检查该网格单元是否有对象
                    if (label[offset + 4] > 0.5f) {
                        // 翻转x坐标（1.0 - x）
                        label[offset + 0] = 1.0f - label[offset + 0];
                    }
                }
            }
        }
    }
    
    // 随机调整亮度、对比度、饱和度
    static void randomColorJitter(std::vector<float>& image, int width, int height, int channels,
                                 float brightness = 0.2f, float contrast = 0.2f, float saturation = 0.2f) {
        // 此处省略实现...
    }
    
    // 随机裁剪
    static void randomCrop(std::vector<float>& image, std::vector<float>& label,
                          int width, int height, int channels,
                          int S, int B, int num_classes) {
        // 此处省略实现...
    }
};
```

## 修改代码支持真实数据集

### 1. 创建数据集加载器

首先，我们需要创建一个数据集加载器类，用于加载和处理真实数据集：

```cpp
// 添加到项目中的新文件：dataset.h
#ifndef YOLOV1_DATASET_H
#define YOLOV1_DATASET_H

#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include "data_loader.h"

class Dataset {
private:
    std::vector<std::vector<float>> images;
    std::vector<std::vector<float>> labels;
    int batch_size;
    int current_index;
    std::vector<int> indices;
    
public:
    Dataset(const std::vector<std::vector<float>>& images,
           const std::vector<std::vector<float>>& labels,
           int batch_size)
        : images(images), labels(labels), batch_size(batch_size), current_index(0) {
        
        // 创建索引数组
        indices.resize(images.size());
        for (size_t i = 0; i < images.size(); i++) {
            indices[i] = static_cast<int>(i);
        }
        
        // 随机打乱索引
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);
    }
    
    // 获取下一个批次
    bool getNextBatch(std::vector<std::vector<float>>& batch_images,
                     std::vector<std::vector<float>>& batch_labels) {
        batch_images.clear();
        batch_labels.clear();
        
        if (current_index >= images.size()) {
            // 一个epoch结束，重新打乱数据
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(indices.begin(), indices.end(), g);
            current_index = 0;
            return false;
        }
        
        // 获取当前批次的数据
        for (int i = 0; i < batch_size && current_index < images.size(); i++, current_index++) {
            int idx = indices[current_index];
            batch_images.push_back(images[idx]);
            batch_labels.push_back(labels[idx]);
        }
        
        return true;
    }
    
    // 获取数据集大小
    size_t size() const {
        return images.size();
    }
    
    // 重置迭代器
    void reset() {
        current_index = 0;
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);
    }
};

#endif // YOLOV1_DATASET_H
```

### 2. 修改训练函数

接下来，我们需要修改YOLODetector类的train函数，使其支持批量训练和更多的训练选项：

```cpp
// 修改src/loss.cpp中的train函数
void YOLODetector::train(const std::vector<std::vector<float>>& training_data,
                       const std::vector<std::vector<float>>& labels,
                       int epochs, float learning_rate,
                       int batch_size, float weight_decay, float momentum) {
    
    // 创建数据集
    Dataset dataset(training_data, labels, batch_size);
    
    // 训练循环
    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        int batch_count = 0;
        
        // 重置数据集迭代器
        dataset.reset();
        
        // 批量训练
        std::vector<std::vector<float>> batch_images;
        std::vector<std::vector<float>> batch_labels;
        
        while (dataset.getNextBatch(batch_images, batch_labels)) {
            float batch_loss = 0.0f;
            
            // 对批次中的每个样本进行训练
            for (size_t i = 0; i < batch_images.size(); i++) {
                // 前向传播
                std::vector<float> predictions = forward(batch_images[i]);
                
                // 计算损失
                float loss = calculateLoss(predictions, batch_labels[i]);
                batch_loss += loss;
                
                // 反向传播和参数更新
                // 这里简化了实现，实际应用中需要计算梯度并更新网络参数
            }
            
            // 计算批次平均损失
            batch_loss /= batch_images.size();
            total_loss += batch_loss;
            batch_count++;
            
            // 打印批次信息（可选）
            std::cout << "Epoch " << epoch + 1 << "/" << epochs
                      << ", Batch " << batch_count
                      << ", Loss: " << batch_loss << std::endl;
        }
        
        // 计算epoch平均损失
        float avg_loss = total_loss / batch_count;
        
        // 打印epoch信息
        std::cout << "Epoch " << epoch + 1 << "/" << epochs
                  << ", Avg Loss: " << avg_loss << std::endl;
        
        // 更新学习率（可选）
        if ((epoch + 1) % 30 == 0) {
            learning_rate *= 0.1f; // 每30个epoch降低学习率
        }
    }
}
```

## 训练配置

### 1. 创建训练配置文件

为了方便配置训练参数，我们可以创建一个配置文件：

```cpp
// 添加到项目中的新文件：config.h
#ifndef YOLOV1_CONFIG_H
#define YOLOV1_CONFIG_H

#include <string>
#include <vector>

struct TrainingConfig {
    // 数据集参数
    std::string dataset_type = "YOLO";  // 数据集类型：YOLO, VOC, COCO
    std::string images_dir;              // 图像目录
    std::string labels_dir;              // 标签目录（YOLO格式）
    std::string annotations_dir;         // 标注目录（VOC格式）
    std::string annotation_file;         // 标注文件（COCO格式）
    
    // 图像参数
    int image_width = 448;
    int image_height = 448;
    int channels = 3;
    
    // 网络参数
    int grid_size = 7;                   // 网格大小
    int boxes_per_cell = 2;              // 每个网格单元预测的边界框数量
    int num_classes = 20;                // 类别数量
    float lambda_coord = 5.0f;           // 坐标损失权重
    float lambda_noobj = 0.5f;           // 无对象损失权重
    
    // 训练参数
    int epochs = 100;                    // 训练轮数
    int batch_size = 64;                 // 批量大小
    float learning_rate = 0.001f;        // 学习率
    float weight_decay = 0.0005f;        // 权重衰减
    float momentum = 0.9f;               // 动量
    
    // 数据增强参数
    bool use_augmentation = true;        // 是否使用数据增强
    float flip_probability = 0.5f;       // 水平翻转概率
    float brightness = 0.2f;             // 亮度调整范围
    float contrast = 0.2f;               // 对比度调整范围
    float saturation = 0.2f;             // 饱和度调整范围
    
    // 模型保存参数
    std::string model_save_path = "yolov1_model.weights"; // 模型保存路径
    int save_interval = 10;              // 保存间隔（每多少个epoch保存一次）
    
    // 类别名称（VOC格式）
    std::vector<std::string> class_names = {
        "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    };
};

#endif // YOLOV1_CONFIG_H
```

### 2. 创建训练脚本

最后，我们创建一个训练脚本，用于加载数据集、配置参数和训练模型：

```cpp
// 添加到项目中的新文件：examples/train_real_data.cpp
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <ctime>
#include "../include/yolo.h"
#include "../include/config.h"
#include "../include/data_loader.h"
#include "../include/dataset.h"

int main(int argc, char** argv) {
    // 设置随机种子
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    
    // 加载配置
    TrainingConfig config;
    
    // 如果提供了配置文件路径，从文件加载配置
    if (argc > 1) {
        std::string config_path = argv[1];
        // 这里应该实现从文件加载配置的功能
        std::cout << "Loading configuration from " << config_path << std::endl;
    }
    
    // 打印配置信息
    std::cout << "YOLOv1 Training on Real Dataset" << std::endl;
    std::cout << "================================" << std::endl;
    std::cout << "Dataset Type: " << config.dataset_type << std::endl;
    std::cout << "Image Size: " << config.image_width << "x" << config.image_height << std::endl;
    std::cout << "Grid Size: " << config.grid_size << "x" << config.grid_size << std::endl;
    std::cout << "Number of Classes: " << config.num_classes << std::endl;
    std::cout << "Batch Size: " << config.batch_size << std::endl;
    std::cout << "Learning Rate: " << config.learning_rate << std::endl;
    std::cout << "Epochs: " << config.epochs << std::endl;
    
    // 加载数据集
    std::cout << "\nLoading dataset..." << std::endl;
    
    std::vector<std::vector<float>> images;
    std::vector<std::vector<float>> labels;
    
    if (config.dataset_type == "YOLO") {
        auto dataset = DataLoader::loadYOLODataset(
            config.images_dir, config.labels_dir,
            config.image_width, config.image_height, config.channels,
            config.grid_size, config.boxes_per_cell, config.num_classes);
        
        images = dataset.first;
        labels = dataset.second;
    }
    else if (config.dataset_type == "VOC") {
        auto dataset = DataLoader::loadVOCDataset(
            config.images_dir, config.annotations_dir,
            config.image_width, config.image_height, config.channels,
            config.grid_size, config.boxes_per_cell, config.num_classes,
            config.class_names);
        
        images = dataset.first;
        labels = dataset.second;
    }
    else if (config.dataset_type == "COCO") {
        auto dataset = DataLoader::loadCOCODataset(
            config.images_dir, config.annotation_file,
            config.image_width, config.image_height, config.channels,
            config.grid_size, config.boxes_per_cell, config.num_classes);
        
        images = dataset.first;
        labels = dataset.second;
    }
    else {
        std::cerr << "Unsupported dataset type: " << config.dataset_type << std::endl;
        return -1;
    }
    
    std::cout << "Loaded " << images.size() << " training samples." << std::endl;
    
    // 创建YOLO检测器
    YOLOParams params(config.grid_size, config.boxes_per_cell, config.num_classes,
                     config.lambda_coord, config.lambda_noobj);
    YOLODetector detector(params);
    
    // 开始训练
    std::cout << "\nStarting training..." << std::endl;
    detector.train(images, labels, config.epochs, config.learning_rate,
                  config.batch_size, config.weight_decay, config.momentum);
    
    // 保存训练好的模型
    if (detector.saveWeights(config.model_save_path)) {
        std::cout << "\nModel saved to " << config.model_save_path << std::endl;
    } else {
        std::cerr << "\nFailed to save model!" << std::endl;
    }
    
    std::cout << "\nTraining completed!" << std::endl;
    
    return 0;
}
```

## 训练过程

### 1. 编译项目

首先，我们需要更新CMakeLists.txt文件，添加新的训练程序：

```cmake
# 添加到CMakeLists.txt

# 创建真实数据集训练可执行文件
add_executable(yolov1_train_real examples/train_real_data.cpp)
target_link_libraries(yolov1_train_real yolov1_lib)

# 如果使用OpenCV，取消下面的注释
# find_package(OpenCV REQUIRED)
# target_link_libraries(yolov1_train_real yolov1_lib ${OpenCV_LIBS})
```

然后编译项目：

```bash
mkdir -p build && cd build
cmake ..
cmake --build .
```

### 2. 准备配置文件

创建一个配置文件，例如`config.txt`，设置训练参数：

```
dataset_type=YOLO
images_dir=/path/to/images
labels_dir=/path/to/labels
grid_size=7
num_classes=20
epochs=100
batch_size=64
learning_rate=0.001
```

### 3. 开始训练

运行训练程序：

```bash
./yolov1_train_real config.txt
```

## 评估模型

训练完成后，我们需要评估模型的性能。以下是一个简单的评估脚本：

```cpp
// 添加到项目中的新文件：examples/evaluate.cpp
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include "../include/yolo.h"
#include "../include/config.h"
#include "../include/data_loader.h"

// 计算AP（平均精度）
float calculateAP(const std::vector<BoundingBox>& detections,
                const std::vector<BoundingBox>& ground_truth,
                float iou_threshold = 0.5) {
    // 此处省略实现...
    return 0.0f;
}

// 计算mAP（平均精度均值）
float calculateMAP(const std::vector<std::vector<BoundingBox>>& all_detections,
                 const std::vector<std::vector<BoundingBox>>& all_ground_truth,
                 int num_classes, float iou_threshold = 0.5) {
    std::vector<float> aps(num_classes, 0.0f);
    
    for (int c = 0; c < num_classes; c++) {
        // 收集该类别的所有检测结果和真实框
        std::vector<BoundingBox> class_detections;
        std::vector<BoundingBox> class_ground_truth;
        
        for (size_t i = 0; i < all_detections.size(); i++) {
            for (const auto& box : all_detections[i]) {
                if (box.class_id == c) {
                    class_detections.push_back(box);
                }
            }
            
            for (const auto& box : all_ground_truth[i]) {
                if (box.class_id == c) {
                    class_ground_truth.push_back(box);
                }
            }
        }
        
        // 计算该类别的AP
        aps[c] = calculateAP(class_detections, class_ground_truth, iou_threshold);
    }
    
    // 计算mAP
    float mAP = 0.0f;
    for (float ap : aps) {
        mAP += ap;
    }
    mAP /= num_classes;
    
    return mAP;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <config_path>" << std::endl;
        return -1;
    }
    
    std::string model_path = argv[1];
    std::string config_path = argv[2];
    
    // 加载配置
    TrainingConfig config;
    // 从文件加载配置...
    
    // 创建YOLO检测器
    YOLOParams params(config.grid_size, config.boxes_per_cell, config.num_classes,
                     config.lambda_coord, config.lambda_noobj);
    YOLODetector detector(params);
    
    // 加载模型
    if (!detector.loadWeights(model_path)) {
        std::cerr << "Failed to load model from " << model_path << std::endl;
        return -1;
    }
    
    std::cout << "Model loaded from " << model_path << std::endl;
    
    // 加载测试数据集
    std::cout << "Loading test dataset..." << std::endl;
    
    std::vector<std::vector<float>> test_images;
    std::vector<std::vector<float>> test_labels;
    
    // 加载测试数据集...
    
    std::cout << "Loaded " << test_images.size() << " test samples." << std::endl;
    
    // 对测试集进行预测
    std::vector<std::vector<BoundingBox>> all_detections;
    std::vector<std::vector<BoundingBox>> all_ground_truth;
    
    for (size_t i = 0; i < test_images.size(); i++) {
        // 前向传播
        std::vector<float> output = detector.forward(test_images[i]);
        
        // 解码输出为边界框
        std::vector<BoundingBox> boxes = detector.decodeOutput(output, 0.1f);
        
        // 应用NMS
        std::vector<BoundingBox> final_boxes = detector.applyNMS(boxes, 0.45f);
        
        // 解码真实标签为边界框
        std::vector<BoundingBox> gt_boxes;
        // 从test_labels[i]解析真实边界框...
        
        all_detections.push_back(final_boxes);
        all_ground_truth.push_back(gt_boxes);
    }
    
    // 计算mAP
    float mAP = calculateMAP(all_detections, all_ground_truth, config.num_classes, 0.5f);
    
    std::cout << "mAP@0.5: " << std::fixed << std::setprecision(4) << mAP << std::endl;
    
    return 0;
}
```

## 实际检测

最后，我们可以使用训练好的模型对新图像进行检测：

```cpp
// 添加到项目中的新文件：examples/detect.cpp
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include "../include/yolo.h"
#include "../include/config.h"

// 加载图像函数（需要使用OpenCV或其他图像处理库）
std::vector<float> loadImage(const std::string& image_path, int width, int height) {
    // 此处省略实现...
    std::vector<float> image_data(width * height * 3, 0.0f);
    return image_data;
}

// 绘制检测结果函数（需要使用OpenCV或其他图像处理库）
void drawDetections(const std::string& image_path, const std::vector<BoundingBox>& boxes,
                   const std::vector<std::string>& class_names,
                   const std::string& output_path) {
    // 此处省略实现...
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <image_path> [output_path]" << std::endl;
        return -1;
    }
    
    std::string model_path = argv[1];
    std::string image_path = argv[2];
    std::string output_path = (argc > 3) ? argv[3] : "output.jpg";
    
    // 创建YOLO检测器
    YOLOParams params(7, 2, 20); // 默认参数
    YOLODetector detector(params);
    
    // 加载模型
    if (!detector.loadWeights(model_path)) {
        std::cerr << "Failed to load model from " << model_path << std::endl;
        return -1;
    }
    
    std::cout << "Model loaded from " << model_path << std::endl;
    
    // 加载图像
    std::vector<float> image = loadImage(image_path, 448, 448);
    
    // 前向传播
    std::vector<float> output = detector.forward(image);
    
    // 解码输出为边界框
    float confidence_threshold = 0.2f;
    std::vector<BoundingBox> boxes = detector.decodeOutput(output, confidence_threshold);
    
    // 应用NMS
    float nms_threshold = 0.45f;
    std::vector<BoundingBox> final_boxes = detector.applyNMS(boxes, nms_threshold);
    
    // 打印检测结果
    std::cout << "Detected " << final_boxes.size() << " objects:" << std::endl;
    
    std::vector<std::string> class_names = {
        "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    };
    
    for (size_t i = 0; i < final_boxes.size(); i++) {
        const auto& box = final_boxes[i];
        std::string class_name = (box.class_id < class_names.size()) ? 
                                class_names[box.class_id] : 
                                "unknown";
        
        std::cout << i + 1 << ". " << class_name 
                  << " (" << std::fixed << std::setprecision(2) << box.confidence * 100.0f << "%)"
                  << " at (" << box.x << ", " << box.y << ")"
                  << " with size " << box.w << "x" << box.h << std::endl;
    }
    
    // 绘制检测结果
    drawDetections(image_path, final_boxes, class_names, output_path);
    
    std::cout << "\nDetection results saved to " << output_path << std::endl;
    
    return 0;
}
```

## 常见问题

### 1. 训练时内存不足

如果训练时遇到内存不足的问题，可以尝试以下解决方案：

- 减小批量大小（batch_size）
- 减小图像尺寸（不推荐，可能影响检测精度）
- 使用数据生成器，按需加载图像，而不是一次性加载所有图像

### 2. 训练时间过长

如果训练时间过长，可以尝试以下解决方案：

- 使用GPU加速训练（需要修改代码以支持GPU）
- 减小训练集大小，先在小数据集上验证模型是否正常工作
- 使用预训练权重初始化模型

### 3. 检测精度低

如果检测精度低，可以尝试以下解决方案：

- 增加训练数据量
- 使用更多的数据增强方法
- 调整网络参数（如lambda_coord和lambda_noobj）
- 使用更复杂的网络结构（如YOLOv2、YOLOv3等）

### 4. 类别不平衡

如果数据集中某些类别的样本数量远少于其他类别，可能导致模型对这些类别的检测性能较差。可以尝试以下解决方案：

- 对少数类别进行过采样
- 对多数类别进行欠采样
- 使用加权损失函数，增加少数类别的权重

### 5. 小目标检测困难

YOLOv1对小目标的检测性能较差，可以尝试以下解决方案：

- 增加小目标的训练样本
- 使用更高分辨率的输入图像
- 考虑使用YOLOv2、YOLOv3等改进版本，它们对小目标的检测性能更好