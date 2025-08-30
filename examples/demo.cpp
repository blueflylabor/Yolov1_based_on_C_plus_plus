#include "../include/yolo.h"
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>

// 打印检测结果
void printDetections(const std::vector<BoundingBox>& detections) {
    std::cout << "\nDetection Results:" << std::endl;
    std::cout << std::setw(5) << "ID" << std::setw(10) << "Class" 
              << std::setw(12) << "Confidence" << std::setw(10) << "X" 
              << std::setw(10) << "Y" << std::setw(10) << "Width" 
              << std::setw(10) << "Height" << std::endl;
    std::cout << std::string(67, '-') << std::endl;
    
    for (size_t i = 0; i < detections.size(); i++) {
        const auto& box = detections[i];
        std::cout << std::setw(5) << i 
                  << std::setw(10) << box.class_id 
                  << std::setw(12) << std::fixed << std::setprecision(4) << box.confidence 
                  << std::setw(10) << std::fixed << std::setprecision(4) << box.x 
                  << std::setw(10) << std::fixed << std::setprecision(4) << box.y 
                  << std::setw(10) << std::fixed << std::setprecision(4) << box.w 
                  << std::setw(10) << std::fixed << std::setprecision(4) << box.h << std::endl;
    }
}

// 创建一个模拟的图像数据
std::vector<float> createDummyImage(int width, int height, int channels) {
    // 创建一个随机图像数据
    std::vector<float> image(width * height * channels, 0.0f);
    for (size_t i = 0; i < image.size(); i++) {
        image[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    return image;
}

// 创建一些模拟的检测结果
std::vector<float> createDummyDetections(int S, int B, int num_classes) {
    int output_size = S * S * (B * 5 + num_classes);
    std::vector<float> detections(output_size, 0.0f);
    
    // 添加一些随机的检测结果
    for (int i = 0; i < S; i++) {
        for (int j = 0; j < S; j++) {
            int grid_index = i * S + j;
            int offset = grid_index * (B * 5 + num_classes);
            
            // 随机选择一些网格单元包含对象
            if (rand() % 10 < 2) { // 20%的概率有对象
                // 为第一个边界框设置参数
                detections[offset + 0] = static_cast<float>(j) / S + 0.1f * (static_cast<float>(rand()) / RAND_MAX - 0.5f); // x
                detections[offset + 1] = static_cast<float>(i) / S + 0.1f * (static_cast<float>(rand()) / RAND_MAX - 0.5f); // y
                detections[offset + 2] = 0.1f + 0.2f * static_cast<float>(rand()) / RAND_MAX; // w
                detections[offset + 3] = 0.1f + 0.2f * static_cast<float>(rand()) / RAND_MAX; // h
                detections[offset + 4] = 0.7f + 0.3f * static_cast<float>(rand()) / RAND_MAX; // confidence
                
                // 随机选择一个类别
                int class_id = rand() % num_classes;
                detections[offset + B * 5 + class_id] = 0.8f + 0.2f * static_cast<float>(rand()) / RAND_MAX;
            }
        }
    }
    
    return detections;
}

int main() {
    std::cout << "YOLOv1 Object Detection Demo" << std::endl;
    std::cout << "===========================" << std::endl;
    
    // 创建YOLO检测器
    YOLOParams params(7, 2, 20); // 7x7网格，每个网格2个边界框，20个类别
    YOLODetector detector(params);
    
    // 创建一个模拟的图像数据
    int image_width = 448;
    int image_height = 448;
    int image_channels = 3;
    std::vector<float> image = createDummyImage(image_width, image_height, image_channels);
    
    std::cout << "\nProcessing image of size " << image_width << "x" << image_height 
              << " with " << image_channels << " channels..." << std::endl;
    
    // 在实际应用中，这里会调用detector.forward(image)进行前向传播
    // 这里我们使用模拟的检测结果
    std::vector<float> network_output = createDummyDetections(params.S, params.B, params.num_classes);
    
    // 解码网络输出为边界框
    float confidence_threshold = 0.5f;
    std::vector<BoundingBox> boxes = detector.decodeOutput(network_output, confidence_threshold);
    
    std::cout << "\nDetected " << boxes.size() << " objects with confidence threshold " 
              << confidence_threshold << std::endl;
    
    // 应用非极大值抑制
    float nms_threshold = 0.4f;
    std::vector<BoundingBox> final_boxes = detector.applyNMS(boxes, nms_threshold);
    
    std::cout << "After NMS: " << final_boxes.size() << " objects with NMS threshold " 
              << nms_threshold << std::endl;
    
    // 打印检测结果
    printDetections(final_boxes);
    
    return 0;
}