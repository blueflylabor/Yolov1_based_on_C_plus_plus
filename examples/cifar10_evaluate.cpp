#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <sstream>
#include "../include/yolo.h"

// 图像尺寸常量
const int IMAGE_WIDTH = 32;
const int IMAGE_HEIGHT = 32;
const int IMAGE_CHANNELS = 3;

// 加载图像为浮点数向量
std::vector<float> loadImageAsVector(const std::string& image_path) {
    // 由于我们将PPM文件重命名为PNG文件，但实际内容仍是PPM格式
    // 所以我们仍然使用PPM格式加载器
    std::ifstream file(image_path, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open image file: " << image_path << std::endl;
        return std::vector<float>();
    }
    
    // 读取PPM头部
    std::string format;
    int width, height, max_val;
    file >> format >> width >> height >> max_val;
    file.get(); // 读取换行符
    
    if (format != "P6" || width != IMAGE_WIDTH || height != IMAGE_HEIGHT) {
        std::cerr << "Error: Unsupported image format or size: " << format << " " << width << "x" << height << std::endl;
        return std::vector<float>();
    }
    
    // 读取像素数据
    std::vector<unsigned char> pixels(width * height * 3);
    file.read(reinterpret_cast<char*>(pixels.data()), pixels.size());
    
    // 转换为浮点数并归一化到[0,1]
    std::vector<float> normalized_pixels(width * height * 3);
    for (size_t i = 0; i < pixels.size(); ++i) {
        normalized_pixels[i] = static_cast<float>(pixels[i]) / max_val;
    }
    
    return normalized_pixels;
}

namespace fs = std::filesystem;

// CIFAR-10类别
const std::vector<std::string> CIFAR10_CLASSES = {
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
};

// 评估指标结构体
struct EvaluationMetrics {
    float precision;
    float recall;
    float f1_score;
    float mAP;
    std::vector<float> class_accuracies;
    float overall_accuracy;
};

// 加载YOLO格式的标签
std::vector<BoundingBox> loadYOLOLabel(const std::string& label_path) {
    std::vector<BoundingBox> boxes;
    std::ifstream file(label_path);
    
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open label file " << label_path << std::endl;
        return boxes;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        int class_id;
        float x_center, y_center, width, height;
        
        if (!(iss >> class_id >> x_center >> y_center >> width >> height)) {
            continue;
        }
        
        BoundingBox box;
        box.x = x_center;
        box.y = y_center;
        box.w = width;
        box.h = height;
        box.confidence = 1.0f;  // 真实标签的置信度设为1
        box.class_id = class_id;
        
        boxes.push_back(box);
    }
    
    file.close();
    return boxes;
}

// 计算IoU (Intersection over Union)
float calculateIoU(const BoundingBox& box1, const BoundingBox& box2) {
    // 计算边界框的左上角和右下角坐标
    float box1_x1 = box1.x - box1.w / 2;
    float box1_y1 = box1.y - box1.h / 2;
    float box1_x2 = box1.x + box1.w / 2;
    float box1_y2 = box1.y + box1.h / 2;
    
    float box2_x1 = box2.x - box2.w / 2;
    float box2_y1 = box2.y - box2.h / 2;
    float box2_x2 = box2.x + box2.w / 2;
    float box2_y2 = box2.y + box2.h / 2;
    
    // 计算交集区域的坐标
    float inter_x1 = std::max(box1_x1, box2_x1);
    float inter_y1 = std::max(box1_y1, box2_y1);
    float inter_x2 = std::min(box1_x2, box2_x2);
    float inter_y2 = std::min(box1_y2, box2_y2);
    
    // 计算交集面积
    float inter_width = std::max(0.0f, inter_x2 - inter_x1);
    float inter_height = std::max(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_width * inter_height;
    
    // 计算两个边界框的面积
    float box1_area = box1.w * box1.h;
    float box2_area = box2.w * box2.h;
    
    // 计算并集面积
    float union_area = box1_area + box2_area - inter_area;
    
    // 计算IoU
    return inter_area / union_area;
}

// 评估模型性能
EvaluationMetrics evaluateModel(YOLODetector& model, const std::string& val_images_dir, const std::string& val_labels_dir, float iou_threshold = 0.5, float conf_threshold = 0.5) {
    EvaluationMetrics metrics;
    metrics.class_accuracies.resize(CIFAR10_CLASSES.size(), 0.0f);
    
    std::vector<int> true_positives(CIFAR10_CLASSES.size(), 0);
    std::vector<int> false_positives(CIFAR10_CLASSES.size(), 0);
    std::vector<int> false_negatives(CIFAR10_CLASSES.size(), 0);
    std::vector<int> class_counts(CIFAR10_CLASSES.size(), 0);
    
    int total_correct = 0;
    int total_predictions = 0;
    
    // 获取验证集图像文件列表
    std::vector<std::string> image_files;
    for (const auto& entry : fs::directory_iterator(val_images_dir)) {
        if (entry.path().extension() == ".png") {
            image_files.push_back(entry.path().string());
        }
    }
    
    for (const auto& image_path : image_files) {
        // 获取对应的标签文件路径
        std::string filename = fs::path(image_path).stem().string();
        std::string label_path = val_labels_dir + "/" + filename + ".txt";
        
        // 加载真实标签
        std::vector<BoundingBox> ground_truth = loadYOLOLabel(label_path);
        if (ground_truth.empty()) {
            continue;
        }
        
        // 更新类别计数
        for (const auto& gt_box : ground_truth) {
            if (gt_box.class_id >= 0 && gt_box.class_id < CIFAR10_CLASSES.size()) {
                class_counts[gt_box.class_id]++;
            }
        }
        
        // 使用模型进行预测
    // 这里我们需要加载图像，进行前向传播，然后解码输出
    // 简化实现：假设我们有一个辅助函数来加载图像
    std::vector<float> image_data = loadImageAsVector(image_path);
    
    // 前向传播
    std::vector<float> output = model.forward(image_data);
    
    // 解码输出为边界框
    std::vector<BoundingBox> predictions = model.decodeOutput(output, conf_threshold);
    
    // 应用非极大值抑制
    predictions = model.applyNMS(predictions, iou_threshold);
        
        // 匹配预测和真实标签
        std::vector<bool> gt_matched(ground_truth.size(), false);
        
        for (const auto& pred_box : predictions) {
            int pred_class = pred_box.class_id;
            bool matched = false;
            
            // 查找最佳匹配的真实标签
            float best_iou = 0.0f;
            int best_gt_idx = -1;
            
            for (size_t i = 0; i < ground_truth.size(); i++) {
                if (ground_truth[i].class_id == pred_class && !gt_matched[i]) {
                    float iou = calculateIoU(pred_box, ground_truth[i]);
                    if (iou > best_iou) {
                        best_iou = iou;
                        best_gt_idx = i;
                    }
                }
            }
            
            // 如果找到匹配且IoU超过阈值
            if (best_gt_idx >= 0 && best_iou >= iou_threshold) {
                true_positives[pred_class]++;
                gt_matched[best_gt_idx] = true;
                matched = true;
                total_correct++;
            } else {
                false_positives[pred_class]++;
            }
            
            total_predictions++;
        }
        
        // 计算未匹配的真实标签（假阴性）
        for (size_t i = 0; i < ground_truth.size(); i++) {
            if (!gt_matched[i]) {
                int gt_class = ground_truth[i].class_id;
                if (gt_class >= 0 && gt_class < CIFAR10_CLASSES.size()) {
                    false_negatives[gt_class]++;
                }
            }
        }
    }
    
    // 计算总体指标
    int total_tp = std::accumulate(true_positives.begin(), true_positives.end(), 0);
    int total_fp = std::accumulate(false_positives.begin(), false_positives.end(), 0);
    int total_fn = std::accumulate(false_negatives.begin(), false_negatives.end(), 0);
    
    // 计算精确度、召回率和F1分数
    metrics.precision = total_tp > 0 ? static_cast<float>(total_tp) / (total_tp + total_fp) : 0.0f;
    metrics.recall = total_tp > 0 ? static_cast<float>(total_tp) / (total_tp + total_fn) : 0.0f;
    metrics.f1_score = (metrics.precision + metrics.recall > 0) ? 
                      2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall) : 0.0f;
    
    // 计算每个类别的准确率
    float sum_ap = 0.0f;
    for (size_t i = 0; i < CIFAR10_CLASSES.size(); i++) {
        if (class_counts[i] > 0) {
            float precision = (true_positives[i] + false_positives[i] > 0) ? 
                            static_cast<float>(true_positives[i]) / (true_positives[i] + false_positives[i]) : 0.0f;
            float recall = (true_positives[i] + false_negatives[i] > 0) ? 
                          static_cast<float>(true_positives[i]) / (true_positives[i] + false_negatives[i]) : 0.0f;
            
            metrics.class_accuracies[i] = (precision + recall > 0) ? 
                                        2 * precision * recall / (precision + recall) : 0.0f;
            sum_ap += metrics.class_accuracies[i];
        }
    }
    
    // 计算mAP（平均精度均值）
    metrics.mAP = sum_ap / CIFAR10_CLASSES.size();
    
    // 计算总体准确率
    metrics.overall_accuracy = total_predictions > 0 ? static_cast<float>(total_correct) / total_predictions : 0.0f;
    
    return metrics;
}

// 打印评估结果
void printEvaluationResults(const EvaluationMetrics& metrics) {
    std::cout << "\nModel Evaluation Results" << std::endl;
    std::cout << "=======================" << std::endl;
    std::cout << "Precision: " << metrics.precision * 100 << "%" << std::endl;
    std::cout << "Recall: " << metrics.recall * 100 << "%" << std::endl;
    std::cout << "F1 Score: " << metrics.f1_score * 100 << "%" << std::endl;
    std::cout << "mAP: " << metrics.mAP * 100 << "%" << std::endl;
    std::cout << "Overall Accuracy: " << metrics.overall_accuracy * 100 << "%" << std::endl;
    
    std::cout << "\nClass-wise Performance:" << std::endl;
    for (size_t i = 0; i < CIFAR10_CLASSES.size(); i++) {
        std::cout << "  " << CIFAR10_CLASSES[i] << ": " << metrics.class_accuracies[i] * 100 << "%" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <model_path> <val_images_dir> <val_labels_dir> [iou_threshold] [conf_threshold]" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    std::string val_images_dir = argv[2];
    std::string val_labels_dir = argv[3];
    
    float iou_threshold = 0.5f;
    float conf_threshold = 0.5f;
    
    if (argc > 4) {
        iou_threshold = std::stof(argv[4]);
    }
    
    if (argc > 5) {
        conf_threshold = std::stof(argv[5]);
    }
    
    std::cout << "Evaluating YOLOv1 model on CIFAR-10 dataset" << std::endl;
    std::cout << "Model: " << model_path << std::endl;
    std::cout << "Validation Images: " << val_images_dir << std::endl;
    std::cout << "Validation Labels: " << val_labels_dir << std::endl;
    std::cout << "IoU Threshold: " << iou_threshold << std::endl;
    std::cout << "Confidence Threshold: " << conf_threshold << std::endl;
    
    // 加载模型 - 使用CIFAR-10的10个类别
    YOLOParams params(7, 2, 10); // S=7, B=2, classes=10 (CIFAR-10)
    YOLODetector model(params);
    
    if (!model.loadWeights(model_path)) {
        std::cerr << "Error: Failed to load model weights from " << model_path << std::endl;
        return 1;
    }
    
    std::cout << "Model loaded successfully. Starting evaluation..." << std::endl;
    
    // 评估模型
    EvaluationMetrics metrics = evaluateModel(model, val_images_dir, val_labels_dir, iou_threshold, conf_threshold);
    
    // 打印评估结果
    printEvaluationResults(metrics);
    
    return 0;
}