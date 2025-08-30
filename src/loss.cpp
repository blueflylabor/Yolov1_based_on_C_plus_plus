#include "../include/yolo.h"

// 计算损失函数
float YOLODetector::calculateLoss(const std::vector<float>& predictions,
                                const std::vector<float>& ground_truth) {
    const int S = params.S;
    const int B = params.B;
    const int num_classes = params.num_classes;
    const float lambda_coord = params.lambda_coord;
    const float lambda_noobj = params.lambda_noobj;
    
    // 网络输出的大小应该是 S*S*(B*5+num_classes)
    int output_size = S * S * (B * 5 + num_classes);
    if (predictions.size() != output_size || ground_truth.size() != output_size) {
        std::cerr << "Error: Predictions or ground truth size mismatch!" << std::endl;
        return -1.0f;
    }
    
    float loss = 0.0f;
    
    // 遍历每个网格单元
    for (int i = 0; i < S; i++) {
        for (int j = 0; j < S; j++) {
            int grid_index = i * S + j;
            int offset = grid_index * (B * 5 + num_classes);
            
            // 检查该网格单元是否有对象
            bool has_object = false;
            for (int b = 0; b < B; b++) {
                if (ground_truth[offset + b * 5 + 4] > 0.5f) {
                    has_object = true;
                    break;
                }
            }
            
            if (has_object) {
                // 找出与真实框IOU最高的预测框
                int best_box = 0;
                float best_iou = 0.0f;
                
                // 构建真实边界框
                BoundingBox gt_box;
                gt_box.x = ground_truth[offset + 0];
                gt_box.y = ground_truth[offset + 1];
                gt_box.w = ground_truth[offset + 2];
                gt_box.h = ground_truth[offset + 3];
                
                for (int b = 0; b < B; b++) {
                    // 构建预测边界框
                    BoundingBox pred_box;
                    pred_box.x = predictions[offset + b * 5 + 0];
                    pred_box.y = predictions[offset + b * 5 + 1];
                    pred_box.w = predictions[offset + b * 5 + 2];
                    pred_box.h = predictions[offset + b * 5 + 3];
                    
                    // 计算IOU
                    float iou = pred_box.calculateIOU(gt_box);
                    if (iou > best_iou) {
                        best_iou = iou;
                        best_box = b;
                    }
                }
                
                // 计算坐标损失（使用平方误差）
                float x_loss = std::pow(predictions[offset + best_box * 5 + 0] - ground_truth[offset + 0], 2);
                float y_loss = std::pow(predictions[offset + best_box * 5 + 1] - ground_truth[offset + 1], 2);
                float w_loss = std::pow(std::sqrt(predictions[offset + best_box * 5 + 2]) - std::sqrt(ground_truth[offset + 2]), 2);
                float h_loss = std::pow(std::sqrt(predictions[offset + best_box * 5 + 3]) - std::sqrt(ground_truth[offset + 3]), 2);
                
                // 坐标损失乘以lambda_coord
                loss += lambda_coord * (x_loss + y_loss + w_loss + h_loss);
                
                // 计算置信度损失（有对象）
                float conf_loss = std::pow(predictions[offset + best_box * 5 + 4] - 1.0f, 2);
                loss += conf_loss;
                
                // 计算其他框的置信度损失（无对象）
                for (int b = 0; b < B; b++) {
                    if (b != best_box) {
                        float conf_loss_noobj = std::pow(predictions[offset + b * 5 + 4] - 0.0f, 2);
                        loss += lambda_noobj * conf_loss_noobj;
                    }
                }
                
                // 计算类别损失
                for (int c = 0; c < num_classes; c++) {
                    float class_loss = std::pow(predictions[offset + B * 5 + c] - ground_truth[offset + B * 5 + c], 2);
                    loss += class_loss;
                }
            } else {
                // 无对象，只计算置信度损失
                for (int b = 0; b < B; b++) {
                    float conf_loss_noobj = std::pow(predictions[offset + b * 5 + 4] - 0.0f, 2);
                    loss += lambda_noobj * conf_loss_noobj;
                }
            }
        }
    }
    
    return loss;
}

// 训练函数
void YOLODetector::train(const std::vector<std::vector<float>>& training_data,
                       const std::vector<std::vector<float>>& labels,
                       int epochs, float learning_rate) {
    // 简化的训练实现
    // 在实际应用中，需要使用优化器（如SGD、Adam等）来更新网络参数
    
    if (training_data.size() != labels.size()) {
        std::cerr << "Error: Training data and labels size mismatch!" << std::endl;
        return;
    }
    
    // 训练循环
    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        
        // 遍历每个训练样本
        for (size_t i = 0; i < training_data.size(); i++) {
            // 前向传播
            std::vector<float> predictions = forward(training_data[i]);
            
            // 计算损失
            float loss = calculateLoss(predictions, labels[i]);
            total_loss += loss;
            
            // 反向传播和参数更新
            // 这里简化了实现，实际应用中需要计算梯度并更新网络参数
        }
        
        // 打印训练信息
        float avg_loss = total_loss / training_data.size();
        std::cout << "Epoch " << epoch + 1 << "/" << epochs << ", Loss: " << avg_loss << std::endl;
        
        // 更新学习率（可选）
        // learning_rate *= 0.95f; // 学习率衰减
    }
}