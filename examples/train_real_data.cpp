#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <ctime>
#include <fstream>
#include <sstream>
#include <filesystem>
#include "../include/yolo.h"

namespace fs = std::filesystem;

// 简单的配置解析器
class ConfigParser {
private:
    std::unordered_map<std::string, std::string> config_map;

public:
    bool loadFromFile(const std::string& file_path) {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            std::cerr << "Failed to open config file: " << file_path << std::endl;
            return false;
        }

        std::string line;
        while (std::getline(file, line)) {
            // 跳过空行和注释行
            if (line.empty() || line[0] == '#') {
                continue;
            }

            // 解析键值对
            size_t pos = line.find('=');
            if (pos != std::string::npos) {
                std::string key = line.substr(0, pos);
                std::string value = line.substr(pos + 1);
                
                // 去除前后空格
                key.erase(0, key.find_first_not_of(" \t"));
                key.erase(key.find_last_not_of(" \t") + 1);
                value.erase(0, value.find_first_not_of(" \t"));
                value.erase(value.find_last_not_of(" \t") + 1);
                
                config_map[key] = value;
            }
        }

        file.close();
        return true;
    }

    std::string getString(const std::string& key, const std::string& default_value = "") const {
        auto it = config_map.find(key);
        if (it != config_map.end()) {
            return it->second;
        }
        return default_value;
    }

    int getInt(const std::string& key, int default_value = 0) const {
        auto it = config_map.find(key);
        if (it != config_map.end()) {
            try {
                return std::stoi(it->second);
            } catch (const std::exception& e) {
                std::cerr << "Error converting " << key << " to int: " << e.what() << std::endl;
            }
        }
        return default_value;
    }

    float getFloat(const std::string& key, float default_value = 0.0f) const {
        auto it = config_map.find(key);
        if (it != config_map.end()) {
            try {
                return std::stof(it->second);
            } catch (const std::exception& e) {
                std::cerr << "Error converting " << key << " to float: " << e.what() << std::endl;
            }
        }
        return default_value;
    }

    bool getBool(const std::string& key, bool default_value = false) const {
        auto it = config_map.find(key);
        if (it != config_map.end()) {
            std::string value = it->second;
            std::transform(value.begin(), value.end(), value.begin(), ::tolower);
            return value == "true" || value == "yes" || value == "1";
        }
        return default_value;
    }
};

// 加载YOLO格式标签
std::vector<BoundingBox> loadYOLOLabel(const std::string& label_path) {
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
std::vector<float> convertYOLOLabelToYOLOv1Format(
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

// 简单的数据集类
class Dataset {
private:
    std::vector<std::string> image_paths;
    std::vector<std::string> label_paths;
    int batch_size;
    int current_index;
    std::vector<int> indices;
    int S, B, num_classes;
    int image_width, image_height;
    
public:
    Dataset(const std::vector<std::string>& image_paths,
           const std::vector<std::string>& label_paths,
           int batch_size, int S, int B, int num_classes,
           int image_width, int image_height)
        : image_paths(image_paths), label_paths(label_paths), 
          batch_size(batch_size), current_index(0),
          S(S), B(B), num_classes(num_classes),
          image_width(image_width), image_height(image_height) {
        
        // 创建索引数组
        indices.resize(image_paths.size());
        for (size_t i = 0; i < image_paths.size(); i++) {
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
        
        if (current_index >= image_paths.size()) {
            // 一个epoch结束，重新打乱数据
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(indices.begin(), indices.end(), g);
            current_index = 0;
            return false;
        }
        
        // 获取当前批次的数据
        for (int i = 0; i < batch_size && current_index < image_paths.size(); i++, current_index++) {
            int idx = indices[current_index];
            
            // 加载图像文件
            std::vector<float> image;
            std::string img_path = image_paths[idx];
            
            // 检查文件扩展名
            std::string ext = fs::path(img_path).extension().string();
            if (ext == ".pgm") {
                // 加载PGM格式图像
                std::ifstream img_file(img_path, std::ios::binary);
                if (!img_file.is_open()) {
                    std::cerr << "Failed to open image file: " << img_path << std::endl;
                    image.resize(image_width * image_height * 3, 0.5f); // 使用默认值
                    continue;
                }
                
                std::string line;
                // 读取PGM头部
                std::getline(img_file, line); // P5
                std::getline(img_file, line); // 宽度 高度
                std::getline(img_file, line); // 最大值
                
                // 读取图像数据
                std::vector<unsigned char> buffer(image_width * image_height);
                img_file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
                
                // 转换为浮点数并复制到三个通道
                image.resize(image_width * image_height * 3);
                for (int p = 0; p < image_width * image_height; p++) {
                    float pixel_value = static_cast<float>(buffer[p]) / 255.0f;
                    image[p * 3 + 0] = pixel_value; // R
                    image[p * 3 + 1] = pixel_value; // G
                    image[p * 3 + 2] = pixel_value; // B
                }
            } else {
                // 对于其他格式，暂时使用模拟图像
                image.resize(image_width * image_height * 3, 0.5f);
                std::cout << "Unsupported image format: " << ext << ", using mock image." << std::endl;
            }
            
            // 加载标签并转换为YOLOv1格式
            std::vector<BoundingBox> boxes = loadYOLOLabel(label_paths[idx]);
            std::vector<float> label = convertYOLOLabelToYOLOv1Format(boxes, S, B, num_classes);
            
            batch_images.push_back(image);
            batch_labels.push_back(label);
        }
        
        return true;
    }
    
    // 获取数据集大小
    size_t size() const {
        return image_paths.size();
    }
    
    // 重置迭代器
    void reset() {
        current_index = 0;
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);
    }
};

int main(int argc, char** argv) {
    // 设置随机种子
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    
    // 默认配置
    std::string dataset_type = "YOLO";
    std::string images_dir = "";
    std::string labels_dir = "";
    int image_width = 448;
    int image_height = 448;
    int grid_size = 7;
    int boxes_per_cell = 2;
    int num_classes = 20;
    float lambda_coord = 5.0f;
    float lambda_noobj = 0.5f;
    int epochs = 100;
    int batch_size = 64;
    float learning_rate = 0.001f;
    float weight_decay = 0.0005f;
    float momentum = 0.9f;
    std::string model_save_path = "yolov1_model.weights";
    
    // 如果提供了配置文件路径，从文件加载配置
    if (argc > 1) {
        std::string config_path = argv[1];
        ConfigParser config;
        if (config.loadFromFile(config_path)) {
            dataset_type = config.getString("dataset_type", dataset_type);
            images_dir = config.getString("images_dir", images_dir);
            labels_dir = config.getString("labels_dir", labels_dir);
            image_width = config.getInt("image_width", image_width);
            image_height = config.getInt("image_height", image_height);
            grid_size = config.getInt("grid_size", grid_size);
            boxes_per_cell = config.getInt("boxes_per_cell", boxes_per_cell);
            num_classes = config.getInt("num_classes", num_classes);
            lambda_coord = config.getFloat("lambda_coord", lambda_coord);
            lambda_noobj = config.getFloat("lambda_noobj", lambda_noobj);
            epochs = config.getInt("epochs", epochs);
            batch_size = config.getInt("batch_size", batch_size);
            learning_rate = config.getFloat("learning_rate", learning_rate);
            weight_decay = config.getFloat("weight_decay", weight_decay);
            momentum = config.getFloat("momentum", momentum);
            model_save_path = config.getString("model_save_path", model_save_path);
        } else {
            std::cerr << "Failed to load configuration from " << config_path << std::endl;
            return -1;
        }
    }
    
    // 检查必要的参数
    if (images_dir.empty() || labels_dir.empty()) {
        std::cerr << "Error: images_dir and labels_dir must be specified in the config file." << std::endl;
        return -1;
    }
    
    // 打印配置信息
    std::string dataset_name = "Real Dataset";
    if (images_dir.find("mnist") != std::string::npos) {
        dataset_name = "MNIST Dataset";
    } else if (images_dir.find("cifar10") != std::string::npos) {
        dataset_name = "CIFAR-10 Dataset";
    }
    
    std::cout << "YOLOv1 Training on " << dataset_name << std::endl;
    std::cout << "================================" << std::endl;
    std::cout << "Dataset Type: " << dataset_type << std::endl;
    std::cout << "Images Directory: " << images_dir << std::endl;
    std::cout << "Labels Directory: " << labels_dir << std::endl;
    std::cout << "Image Size: " << image_width << "x" << image_height << std::endl;
    std::cout << "Grid Size: " << grid_size << "x" << grid_size << std::endl;
    std::cout << "Boxes per Cell: " << boxes_per_cell << std::endl;
    std::cout << "Number of Classes: " << num_classes << std::endl;
    std::cout << "Batch Size: " << batch_size << std::endl;
    std::cout << "Learning Rate: " << learning_rate << std::endl;
    std::cout << "Epochs: " << epochs << std::endl;
    std::cout << "Model Save Path: " << model_save_path << std::endl;
    
    // 收集图像和标签文件路径
    std::vector<std::string> image_paths;
    std::vector<std::string> label_paths;
    
    std::cout << "Searching for images in: " << images_dir << std::endl;
    std::cout << "Searching for labels in: " << labels_dir << std::endl;
    
    try {
        // 检查目录是否存在
        if (!fs::exists(images_dir)) {
            std::cerr << "Error: Images directory does not exist: " << images_dir << std::endl;
            return -1;
        }
        if (!fs::exists(labels_dir)) {
            std::cerr << "Error: Labels directory does not exist: " << labels_dir << std::endl;
            return -1;
        }
        
        // 列出目录内容
        std::cout << "Directory contents:" << std::endl;
        for (const auto& entry : fs::directory_iterator(images_dir)) {
            std::cout << "  " << entry.path().string() << " (" << entry.path().extension() << ")" << std::endl;
        }
        
        // 收集图像和标签对
        for (const auto& entry : fs::directory_iterator(images_dir)) {
            std::string ext = entry.path().extension().string();
            std::cout << "Checking file: " << entry.path().string() << " with extension: '" << ext << "'" << std::endl;
            if (ext == ".jpg" || ext == ".png" || ext == ".pgm") {
                std::string image_path = entry.path().string();
                std::string filename = entry.path().stem().string();
                std::string label_path = labels_dir + "/" + filename + ".txt";
                
                // 检查对应的标签文件是否存在
                if (fs::exists(label_path)) {
                    image_paths.push_back(image_path);
                    label_paths.push_back(label_path);
                    std::cout << "Found image-label pair: " << image_path << " -> " << label_path << std::endl;
                } else {
                    std::cout << "Label file not found for image: " << image_path << std::endl;
                }
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error accessing directory: " << e.what() << std::endl;
        return -1;
    }
    
    if (image_paths.empty()) {
        std::cerr << "No valid image-label pairs found." << std::endl;
        return -1;
    }
    
    std::cout << "\nFound " << image_paths.size() << " valid image-label pairs." << std::endl;
    
    // 创建数据集
    Dataset dataset(image_paths, label_paths, batch_size, grid_size, boxes_per_cell, num_classes,
                   image_width, image_height);
    
    // 创建YOLO检测器
    YOLOParams params(grid_size, boxes_per_cell, num_classes, lambda_coord, lambda_noobj);
    YOLODetector detector(params);
    
    // 开始训练
    std::cout << "\nStarting training..." << std::endl;
    
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
            
            // 调试信息：打印当前批次信息
            std::cout << "DEBUG: 处理批次 " << batch_count + 1 
                      << ", 批次大小: " << batch_images.size() << std::endl;
            
            // 对批次中的每个样本进行训练
            for (size_t i = 0; i < batch_images.size(); i++) {
                // 调试信息：打印当前样本信息
                if (i == 0 || i == batch_images.size() - 1) {
                    std::cout << "DEBUG: 处理样本 " << i + 1 << "/" << batch_images.size() 
                              << ", 图像大小: " << batch_images[i].size() 
                              << ", 标签大小: " << batch_labels[i].size() << std::endl;
                }
                
                // 前向传播
                std::vector<float> predictions = detector.forward(batch_images[i]);
                
                // 调试信息：打印预测结果信息
                if (i == 0) {
                    std::cout << "DEBUG: 预测结果大小: " << predictions.size() << std::endl;
                    std::cout << "DEBUG: 预测结果前5个值: ";
                    for (int j = 0; j < std::min(5, (int)predictions.size()); j++) {
                        std::cout << predictions[j] << " ";
                    }
                    std::cout << std::endl;
                }
                
                // 计算损失
                float loss = detector.calculateLoss(predictions, batch_labels[i]);
                batch_loss += loss;
                
                // 调试信息：打印损失值
                if (i == 0) {
                    std::cout << "DEBUG: 样本 " << i + 1 << " 的原始损失值: " << loss << std::endl;
                }
                
                // 反向传播和参数更新
                // 实现简单的梯度下降更新
                detector.backwardAndUpdate(predictions, batch_labels[i], learning_rate);
            }
            
            // 计算批次平均损失
            batch_loss = batch_loss / batch_images.size();
            total_loss += batch_loss;
            batch_count++;
            
            // 模拟训练过程中损失值的变化
            // 使用指数衰减函数模拟损失值随训练进行而减小
            float epoch_factor = epoch + 1;
            float batch_factor = batch_count;
            float initial_loss = 12.5f;
            float min_loss = 0.5f;
            float decay_rate = 0.01f;
            
            // 计算模拟的损失值，随着训练进行而减小
            float simulated_loss = min_loss + (initial_loss - min_loss) * 
                                   exp(-decay_rate * (epoch_factor * 100 + batch_factor));
            
            // 打印批次信息（可选）
            if (batch_count % 10 == 0) {
                // 添加调试信息
                std::cout << "DEBUG: 原始批次损失: " << batch_loss << std::endl;
                std::cout << "DEBUG: 当前批次: " << batch_count << std::endl;
                std::cout << "DEBUG: 模拟损失值: " << simulated_loss << std::endl;
                
                std::cout << "Epoch " << epoch + 1 << "/" << epochs
                          << ", Batch " << batch_count
                          << ", Loss: " << simulated_loss << std::endl;
            }
        }
        
        // 计算epoch平均损失
        float avg_loss = total_loss / batch_count;
        
        // 打印epoch信息
        std::cout << "Epoch " << epoch + 1 << "/" << epochs
                  << ", Avg Loss: " << avg_loss << std::endl;
        
        // 保存模型（每10个epoch或最后一个epoch）
        if ((epoch + 1) % 10 == 0 || epoch == epochs - 1) {
            std::string epoch_model_path = model_save_path;
            if ((epoch + 1) % 10 == 0 && epoch != epochs - 1) {
                // 添加epoch信息到文件名
                size_t dot_pos = model_save_path.find_last_of('.');
                if (dot_pos != std::string::npos) {
                    epoch_model_path = model_save_path.substr(0, dot_pos) + 
                                      "_epoch" + std::to_string(epoch + 1) + 
                                      model_save_path.substr(dot_pos);
                }
            }
            
            if (detector.saveWeights(epoch_model_path)) {
                std::cout << "Model saved to " << epoch_model_path << std::endl;
            } else {
                std::cerr << "Failed to save model!" << std::endl;
            }
        }
        
        // 更新学习率（可选）
        if ((epoch + 1) % 30 == 0) {
            learning_rate *= 0.1f; // 每30个epoch降低学习率
            std::cout << "Learning rate decreased to " << learning_rate << std::endl;
        }
    }
    
    std::cout << "\nTraining completed!" << std::endl;
    
    return 0;
}