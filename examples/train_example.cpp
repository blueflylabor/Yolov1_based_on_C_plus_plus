#include "../include/yolo.h"
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <ctime>

// 创建模拟的训练数据
std::vector<std::vector<float>> createTrainingData(int num_samples, int width, int height, int channels) {
    std::vector<std::vector<float>> training_data;
    
    std::mt19937 rng(static_cast<unsigned int>(time(nullptr)));
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    for (int i = 0; i < num_samples; i++) {
        std::vector<float> image(width * height * channels);
        for (size_t j = 0; j < image.size(); j++) {
            image[j] = dist(rng);
        }
        training_data.push_back(image);
    }
    
    return training_data;
}

// 创建模拟的标签数据
std::vector<std::vector<float>> createLabels(int num_samples, int S, int B, int num_classes) {
    std::vector<std::vector<float>> labels;
    
    std::mt19937 rng(static_cast<unsigned int>(time(nullptr)));
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::uniform_int_distribution<int> grid_dist(0, S-1);
    std::uniform_int_distribution<int> class_dist(0, num_classes-1);
    
    int output_size = S * S * (B * 5 + num_classes);
    
    for (int i = 0; i < num_samples; i++) {
        std::vector<float> label(output_size, 0.0f);
        
        // 每个样本随机放置1-3个对象
        int num_objects = 1 + rand() % 3;
        
        for (int obj = 0; obj < num_objects; obj++) {
            // 随机选择网格位置
            int grid_i = grid_dist(rng);
            int grid_j = grid_dist(rng);
            int grid_index = grid_i * S + grid_j;
            int offset = grid_index * (B * 5 + num_classes);
            
            // 设置边界框参数（相对于网格单元）
            float x_offset = dist(rng) * 0.5f;
            float y_offset = dist(rng) * 0.5f;
            
            // x, y 是相对于整个图像的坐标（0-1范围）
            label[offset + 0] = (grid_j + x_offset) / S; // x
            label[offset + 1] = (grid_i + y_offset) / S; // y
            label[offset + 2] = 0.1f + 0.2f * dist(rng); // w
            label[offset + 3] = 0.1f + 0.2f * dist(rng); // h
            
            // 对于第一个边界框，设置置信度为1
            label[offset + 4] = 1.0f;
            
            // 随机选择一个类别
            int class_id = class_dist(rng);
            label[offset + B * 5 + class_id] = 1.0f;
        }
        
        labels.push_back(label);
    }
    
    return labels;
}

int main() {
    std::cout << "YOLOv1 Training Example" << std::endl;
    std::cout << "======================" << std::endl;
    
    // 设置随机种子
    srand(static_cast<unsigned int>(time(nullptr)));
    
    // 创建YOLO检测器
    YOLOParams params(7, 2, 20); // 7x7网格，每个网格2个边界框，20个类别
    YOLODetector detector(params);
    
    // 训练参数
    int num_samples = 100;    // 训练样本数量
    int image_width = 448;    // 图像宽度
    int image_height = 448;   // 图像高度
    int image_channels = 3;   // 图像通道数
    int epochs = 10;          // 训练轮数
    float learning_rate = 0.001f; // 学习率
    
    std::cout << "\nCreating training data..." << std::endl;
    
    // 创建模拟的训练数据和标签
    std::vector<std::vector<float>> training_data = createTrainingData(
        num_samples, image_width, image_height, image_channels);
    
    std::vector<std::vector<float>> labels = createLabels(
        num_samples, params.S, params.B, params.num_classes);
    
    std::cout << "Created " << training_data.size() << " training samples." << std::endl;
    
    // 开始训练
    std::cout << "\nStarting training..." << std::endl;
    detector.train(training_data, labels, epochs, learning_rate);
    
    // 保存训练好的模型
    std::string model_path = "yolov1_model.weights";
    if (detector.saveWeights(model_path)) {
        std::cout << "\nModel saved to " << model_path << std::endl;
    } else {
        std::cerr << "\nFailed to save model!" << std::endl;
    }
    
    std::cout << "\nTraining completed!" << std::endl;
    
    // 测试训练好的模型
    std::cout << "\nTesting model on a sample image..." << std::endl;
    
    // 使用第一个训练样本进行测试
    std::vector<float> test_image = training_data[0];
    
    // 前向传播
    std::vector<float> network_output = detector.forward(test_image);
    
    // 解码网络输出为边界框
    float confidence_threshold = 0.5f;
    std::vector<BoundingBox> boxes = detector.decodeOutput(network_output, confidence_threshold);
    
    std::cout << "Detected " << boxes.size() << " objects with confidence threshold " 
              << confidence_threshold << std::endl;
    
    // 应用非极大值抑制
    float nms_threshold = 0.4f;
    std::vector<BoundingBox> final_boxes = detector.applyNMS(boxes, nms_threshold);
    
    std::cout << "After NMS: " << final_boxes.size() << " objects" << std::endl;
    
    // 打印检测结果
    if (!final_boxes.empty()) {
        std::cout << "\nDetection Results:" << std::endl;
        for (size_t i = 0; i < final_boxes.size(); i++) {
            const auto& box = final_boxes[i];
            std::cout << "Object " << i << ": Class=" << box.class_id 
                      << ", Confidence=" << box.confidence 
                      << ", Position=" << box.x << "," << box.y 
                      << ", Size=" << box.w << "x" << box.h << std::endl;
        }
    }
    
    return 0;
}