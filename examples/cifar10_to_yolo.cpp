#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <random>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <unistd.h> // 用于getcwd函数
#include "../include/yolo.h"

namespace fs = std::filesystem;

// CIFAR-10类别
const std::vector<std::string> CIFAR10_CLASSES = {
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
};

// CIFAR-10数据结构
struct CIFAR10_Image {
    uint8_t label;           // 标签 (0-9)
    uint8_t data[3072];     // 32x32x3 图像数据
};

// 加载CIFAR-10二进制文件
std::vector<CIFAR10_Image> loadCIFAR10Batch(const std::string& filename) {
    std::vector<CIFAR10_Image> images;
    
    // 打印当前工作目录和尝试打开的文件路径，帮助调试
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        std::cout << "Current working directory: " << cwd << std::endl;
    }
    std::cout << "Trying to open file: " << filename << std::endl;
    
    // 检查文件是否存在
    if (!fs::exists(filename)) {
        std::cerr << "Error: File does not exist: " << filename << std::endl;
        return images;
    }
    
    std::ifstream file(filename, std::ios::binary);
    
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return images;
    }
    
    // 每个CIFAR-10批次文件包含10000张图像
    const int num_images = 10000;
    images.resize(num_images);
    
    for (int i = 0; i < num_images; i++) {
        // 读取标签 (1字节)
        file.read(reinterpret_cast<char*>(&images[i].label), 1);
        
        // 读取图像数据 (3072字节 = 32x32x3)
        file.read(reinterpret_cast<char*>(images[i].data), 3072);
        
        if (file.fail()) {
            std::cerr << "Error: Failed to read image " << i << " from " << filename << std::endl;
            images.resize(i);
            break;
        }
    }
    
    file.close();
    return images;
}

// 将CIFAR-10图像保存为PPM文件（一种简单的图像格式，不需要额外的库）
bool saveCIFAR10ImageAsPNG(const CIFAR10_Image& image, const std::string& output_path) {
    // 使用PPM格式保存图像，这是一种简单的无压缩图像格式
    // 不需要额外的库，可以直接用标准C++实现
    std::ofstream file(output_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot create image file " << output_path << std::endl;
        return false;
    }
    
    // PPM头部信息
    file << "P6\n";
    file << "32 32\n";
    file << "255\n";
    
    // 写入图像数据
    // CIFAR-10数据格式是按通道存储的（所有R，然后所有G，然后所有B）
    // 需要转换为交错格式（RGBRGBRGB...）
    for (int y = 0; y < 32; y++) {
        for (int x = 0; x < 32; x++) {
            int idx = y * 32 + x;
            // 红色通道
            file.put(image.data[idx]);
            // 绿色通道
            file.put(image.data[idx + 1024]);
            // 蓝色通道
            file.put(image.data[idx + 2048]);
        }
    }
    
    file.close();
    return true;
}

// 创建YOLO格式的标签文件
bool createYOLOLabel(const CIFAR10_Image& image, const std::string& output_path) {
    std::ofstream file(output_path);
    if (!file.is_open()) {
        return false;
    }
    
    // CIFAR-10图像中的对象通常占据整个图像
    // 因此，我们将边界框设置为图像中心，宽度和高度为图像的80%
    // 格式：<class_id> <x_center> <y_center> <width> <height>
    file << static_cast<int>(image.label) << " 0.5 0.5 0.8 0.8" << std::endl;
    
    file.close();
    return true;
}

// 将CIFAR-10数据集转换为YOLO格式
bool convertCIFAR10ToYOLO(const std::string& cifar10_dir, const std::string& output_dir) {
    // 创建输出目录
    fs::create_directories(output_dir + "/images/train");
    fs::create_directories(output_dir + "/images/val");
    fs::create_directories(output_dir + "/labels/train");
    fs::create_directories(output_dir + "/labels/val");
    
    // 加载训练批次
    std::vector<CIFAR10_Image> train_images;
    for (int i = 1; i <= 5; i++) {
        std::string batch_file = cifar10_dir + "/data_batch_" + std::to_string(i) + ".bin";
        auto batch = loadCIFAR10Batch(batch_file);
        train_images.insert(train_images.end(), batch.begin(), batch.end());
    }
    
    // 加载测试批次
    std::string test_batch_file = cifar10_dir + "/test_batch.bin";
    auto test_images = loadCIFAR10Batch(test_batch_file);
    
    // 处理训练图像
    for (size_t i = 0; i < train_images.size(); i++) {
        std::string image_path = output_dir + "/images/train/train_" + std::to_string(i) + ".png";
        std::string label_path = output_dir + "/labels/train/train_" + std::to_string(i) + ".txt";
        
        if (!saveCIFAR10ImageAsPNG(train_images[i], image_path)) {
            std::cerr << "Error: Failed to save training image " << i << std::endl;
            continue;
        }
        
        if (!createYOLOLabel(train_images[i], label_path)) {
            std::cerr << "Error: Failed to create training label " << i << std::endl;
            continue;
        }
    }
    
    // 处理测试图像
    for (size_t i = 0; i < test_images.size(); i++) {
        std::string image_path = output_dir + "/images/val/val_" + std::to_string(i) + ".png";
        std::string label_path = output_dir + "/labels/val/val_" + std::to_string(i) + ".txt";
        
        if (!saveCIFAR10ImageAsPNG(test_images[i], image_path)) {
            std::cerr << "Error: Failed to save validation image " << i << std::endl;
            continue;
        }
        
        if (!createYOLOLabel(test_images[i], label_path)) {
            std::cerr << "Error: Failed to create validation label " << i << std::endl;
            continue;
        }
    }
    
    std::cout << "Converted " << train_images.size() << " training images and "
              << test_images.size() << " validation images to YOLO format." << std::endl;
    
    return true;
}

// 创建CIFAR-10训练配置文件
bool createCIFAR10Config(const std::string& output_dir) {
    std::string config_path = output_dir + "/cifar10_config.txt";
    std::ofstream file(config_path);
    
    if (!file.is_open()) {
        std::cerr << "Error: Cannot create config file " << config_path << std::endl;
        return false;
    }
    
    file << "# CIFAR-10训练配置文件\n";
    file << "dataset_type=YOLO\n";
    file << "images_dir=" << output_dir << "/images/train\n";
    file << "labels_dir=" << output_dir << "/labels/train\n";
    file << "\n";
    file << "# 图像参数\n";
    file << "image_width=448\n";
    file << "image_height=448\n";
    file << "\n";
    file << "# 网络参数\n";
    file << "grid_size=7\n";
    file << "boxes_per_cell=2\n";
    file << "num_classes=10\n";  // CIFAR-10有10个类别
    file << "lambda_coord=5.0\n";
    file << "lambda_noobj=0.5\n";
    file << "\n";
    file << "# 训练参数\n";
    file << "epochs=100\n";
    file << "batch_size=64\n";
    file << "learning_rate=0.001\n";
    file << "weight_decay=0.0005\n";
    file << "momentum=0.9\n";
    file << "\n";
    file << "# 模型保存参数\n";
    file << "model_save_path=cifar10_model.weights\n";
    
    file.close();
    std::cout << "Created CIFAR-10 config file: " << config_path << std::endl;
    
    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <cifar10_dir> <output_dir>" << std::endl;
        std::cout << "  cifar10_dir: Directory containing CIFAR-10 binary files" << std::endl;
        std::cout << "  output_dir: Directory to save converted YOLO format files" << std::endl;
        return 1;
    }
    
    std::string cifar10_dir = argv[1];
    std::string output_dir = argv[2];
    
    std::cout << "Converting CIFAR-10 dataset to YOLO format..." << std::endl;
    std::cout << "CIFAR-10 directory: " << cifar10_dir << std::endl;
    std::cout << "Output directory: " << output_dir << std::endl;
    
    // 转换数据集
    if (!convertCIFAR10ToYOLO(cifar10_dir, output_dir)) {
        std::cerr << "Error: Failed to convert CIFAR-10 dataset" << std::endl;
        return 1;
    }
    
    // 创建配置文件
    if (!createCIFAR10Config(output_dir)) {
        std::cerr << "Error: Failed to create CIFAR-10 config file" << std::endl;
        return 1;
    }
    
    std::cout << "\nConversion completed successfully!" << std::endl;
    std::cout << "To train YOLOv1 on CIFAR-10, run:\n" << std::endl;
    std::cout << "  ./yolov1_train_real " << output_dir << "/cifar10_config.txt" << std::endl;
    
    return 0;
}