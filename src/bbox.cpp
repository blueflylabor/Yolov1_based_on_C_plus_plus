#include "../include/yolo.h"

// 解码YOLO输出为边界框
std::vector<BoundingBox> YOLODetector::decodeOutput(const std::vector<float>& network_output,
                                                float confidence_threshold) {
    std::vector<BoundingBox> boxes;
    const int S = params.S;
    const int B = params.B;
    const int num_classes = params.num_classes;
    
    // 网络输出的大小应该是 S*S*(B*5+num_classes)
    // 每个网格单元预测B个边界框，每个边界框有5个参数(x,y,w,h,confidence)
    // 加上num_classes个类别概率
    int output_size = S * S * (B * 5 + num_classes);
    if (network_output.size() != output_size) {
        std::cerr << "Error: Network output size mismatch!" << std::endl;
        return boxes;
    }
    
    // 遍历每个网格单元
    for (int i = 0; i < S; i++) {
        for (int j = 0; j < S; j++) {
            int grid_index = i * S + j;
            
            // 获取类别概率
            std::vector<float> class_probs(num_classes);
            for (int c = 0; c < num_classes; c++) {
                class_probs[c] = network_output[grid_index * (B * 5 + num_classes) + B * 5 + c];
            }
            
            // 遍历每个边界框
            for (int b = 0; b < B; b++) {
                int box_index = grid_index * (B * 5 + num_classes) + b * 5;
                float confidence = network_output[box_index + 4];
                
                // 只处理置信度高于阈值的边界框
                if (confidence > confidence_threshold) {
                    // 获取边界框参数
                    float x = (network_output[box_index + 0] + j) / S;
                    float y = (network_output[box_index + 1] + i) / S;
                    float w = network_output[box_index + 2] * network_output[box_index + 2];
                    float h = network_output[box_index + 3] * network_output[box_index + 3];
                    
                    // 找出最大类别概率
                    float max_class_prob = 0.0f;
                    int max_class_id = -1;
                    for (int c = 0; c < num_classes; c++) {
                        if (class_probs[c] > max_class_prob) {
                            max_class_prob = class_probs[c];
                            max_class_id = c;
                        }
                    }
                    
                    // 创建边界框对象
                    BoundingBox box;
                    box.x = x;
                    box.y = y;
                    box.w = w;
                    box.h = h;
                    box.confidence = confidence;
                    box.class_id = max_class_id;
                    box.class_prob = max_class_prob;
                    
                    boxes.push_back(box);
                }
            }
        }
    }
    
    return boxes;
}

// 应用非极大值抑制
std::vector<BoundingBox> YOLODetector::applyNMS(const std::vector<BoundingBox>& boxes,
                                              float iou_threshold) {
    std::vector<BoundingBox> result;
    if (boxes.empty()) {
        return result;
    }
    
    // 按置信度排序（从高到低）
    std::vector<BoundingBox> sorted_boxes = boxes;
    std::sort(sorted_boxes.begin(), sorted_boxes.end(), 
              [](const BoundingBox& a, const BoundingBox& b) {
                  return a.confidence > b.confidence;
              });
    
    std::vector<bool> is_suppressed(sorted_boxes.size(), false);
    
    // 应用NMS
    for (size_t i = 0; i < sorted_boxes.size(); i++) {
        if (is_suppressed[i]) {
            continue;
        }
        
        result.push_back(sorted_boxes[i]);
        
        // 抑制与当前框IOU高于阈值的其他框
        for (size_t j = i + 1; j < sorted_boxes.size(); j++) {
            if (is_suppressed[j]) {
                continue;
            }
            
            // 如果两个框属于不同类别，不进行抑制
            if (sorted_boxes[i].class_id != sorted_boxes[j].class_id) {
                continue;
            }
            
            // 计算IOU
            float iou = sorted_boxes[i].calculateIOU(sorted_boxes[j]);
            
            // 如果IOU高于阈值，抑制该框
            if (iou > iou_threshold) {
                is_suppressed[j] = true;
            }
        }
    }
    
    return result;
}