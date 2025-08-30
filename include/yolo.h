#ifndef YOLOV1_YOLO_H
#define YOLOV1_YOLO_H

#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <algorithm>
#include <iostream>

// 边界框结构体
struct BoundingBox {
    float x;      // 中心x坐标（相对于图像宽度的比例）
    float y;      // 中心y坐标（相对于图像高度的比例）
    float w;      // 宽度（相对于图像宽度的比例）
    float h;      // 高度（相对于图像高度的比例）
    float confidence; // 置信度
    int class_id;     // 类别ID
    float class_prob; // 类别概率

    // 计算IOU（交并比）
    float calculateIOU(const BoundingBox& other) const {
        // 计算交集区域
        float x1 = std::max(x - w/2, other.x - other.w/2);
        float y1 = std::max(y - h/2, other.y - other.h/2);
        float x2 = std::min(x + w/2, other.x + other.w/2);
        float y2 = std::min(y + h/2, other.y + other.h/2);

        // 计算交集面积
        float intersection_area = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);

        // 计算并集面积
        float box1_area = w * h;
        float box2_area = other.w * other.h;
        float union_area = box1_area + box2_area - intersection_area;

        // 返回IOU
        return intersection_area / union_area;
    }
};

// YOLOv1网络参数
struct YOLOParams {
    int S;              // 网格大小 (SxS)
    int B;              // 每个网格单元预测的边界框数量
    int num_classes;    // 类别数量
    float lambda_coord; // 坐标损失权重
    float lambda_noobj; // 无对象损失权重

    YOLOParams(int grid_size = 7, int boxes_per_cell = 2, int classes = 20,
               float coord_weight = 5.0f, float noobj_weight = 0.5f)
        : S(grid_size), B(boxes_per_cell), num_classes(classes),
          lambda_coord(coord_weight), lambda_noobj(noobj_weight) {}
};

// YOLO检测器类
class YOLODetector {
private:
    YOLOParams params;
    // 网络权重和参数将在实现文件中定义

public:
    // 构造函数
    explicit YOLODetector(const YOLOParams& yolo_params = YOLOParams());

    // 从文件加载模型权重
    bool loadWeights(const std::string& weight_file);

    // 前向传播（推理）
    std::vector<float> forward(const std::vector<float>& input_image);

    // 解码YOLO输出为边界框
    std::vector<BoundingBox> decodeOutput(const std::vector<float>& network_output,
                                         float confidence_threshold = 0.2f);

    // 应用非极大值抑制
    std::vector<BoundingBox> applyNMS(const std::vector<BoundingBox>& boxes,
                                     float iou_threshold = 0.4f);

    // 计算损失函数（用于训练）
    float calculateLoss(const std::vector<float>& predictions,
                       const std::vector<float>& ground_truth);

    // 训练函数
    void train(const std::vector<std::vector<float>>& training_data,
              const std::vector<std::vector<float>>& labels,
              int epochs, float learning_rate);

    // 保存模型权重
    bool saveWeights(const std::string& output_file);
};

#endif // YOLOV1_YOLO_H