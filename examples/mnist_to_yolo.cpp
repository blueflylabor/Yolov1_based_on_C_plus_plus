#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <filesystem>
#include <random>
#include <algorithm>

namespace fs = std::filesystem;

// MNIST文件头结构
struct MNISTHeader {
    uint32_t magic_number;
    uint32_t num_items;
    uint32_t rows;
    uint32_t cols;
};

// 读取大端格式的32位整数
uint32_t readBigEndian(std::ifstream& file) {
    uint8_t buffer[4];
    file.read(reinterpret_cast<char*>(buffer), 4);
    return (buffer[0] << 24) | (buffer[1] << 16) | (buffer[2] << 8) | buffer[3];
}

// 读取MNIST图像文件
bool readMNISTImages(const std::string& filename, std::vector<std::vector<uint8_t>>& images) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "无法打开MNIST图像文件: " << filename << std::endl;
        return false;
    }

    MNISTHeader header;
    header.magic_number = readBigEndian(file);
    header.num_items = readBigEndian(file);
    header.rows = readBigEndian(file);
    header.cols = readBigEndian(file);

    // 检查魔数是否正确 (2051 for images)
    if (header.magic_number != 2051) {
        std::cerr << "MNIST图像文件格式错误" << std::endl;
        return false;
    }

    std::cout << "MNIST图像文件信息:" << std::endl;
    std::cout << "  图像数量: " << header.num_items << std::endl;
    std::cout << "  图像大小: " << header.rows << "x" << header.cols << std::endl;

    // 读取所有图像
    images.resize(header.num_items);
    size_t image_size = header.rows * header.cols;

    for (uint32_t i = 0; i < header.num_items; i++) {
        images[i].resize(image_size);
        file.read(reinterpret_cast<char*>(images[i].data()), image_size);
    }

    file.close();
    return true;
}

// 读取MNIST标签文件
bool readMNISTLabels(const std::string& filename, std::vector<uint8_t>& labels) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "无法打开MNIST标签文件: " << filename << std::endl;
        return false;
    }

    uint32_t magic_number = readBigEndian(file);
    uint32_t num_items = readBigEndian(file);

    // 检查魔数是否正确 (2049 for labels)
    if (magic_number != 2049) {
        std::cerr << "MNIST标签文件格式错误" << std::endl;
        return false;
    }

    std::cout << "MNIST标签文件信息:" << std::endl;
    std::cout << "  标签数量: " << num_items << std::endl;

    // 读取所有标签
    labels.resize(num_items);
    file.read(reinterpret_cast<char*>(labels.data()), num_items);

    file.close();
    return true;
}

// 将MNIST数据转换为YOLO格式
void convertMNISTToYOLO(const std::vector<std::vector<uint8_t>>& images, 
                      const std::vector<uint8_t>& labels,
                      const std::string& output_dir,
                      float train_ratio = 0.8) {
    // 创建输出目录
    fs::create_directories(output_dir + "/images/train");
    fs::create_directories(output_dir + "/images/test");
    fs::create_directories(output_dir + "/labels/train");
    fs::create_directories(output_dir + "/labels/test");

    // 创建索引并随机打乱
    std::vector<size_t> indices(images.size());
    for (size_t i = 0; i < images.size(); i++) {
        indices[i] = i;
    }

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    // 计算训练集和测试集的大小
    size_t train_size = static_cast<size_t>(images.size() * train_ratio);
    size_t test_size = images.size() - train_size;

    std::cout << "划分数据集:" << std::endl;
    std::cout << "  训练集大小: " << train_size << std::endl;
    std::cout << "  测试集大小: " << test_size << std::endl;

    // 处理每个图像
    for (size_t i = 0; i < images.size(); i++) {
        bool is_train = i < train_size;
        std::string set_type = is_train ? "train" : "test";
        std::string image_path = output_dir + "/images/" + set_type + "/" + std::to_string(i) + ".pgm";
        std::string label_path = output_dir + "/labels/" + set_type + "/" + std::to_string(i) + ".txt";

        // 获取原始索引
        size_t idx = indices[i];
        uint8_t label = labels[idx];
        const auto& image = images[idx];

        // 保存图像为PGM格式
        std::ofstream img_file(image_path, std::ios::binary);
        if (img_file.is_open()) {
            // PGM头部
            img_file << "P5\n28 28\n255\n";
            img_file.write(reinterpret_cast<const char*>(image.data()), image.size());
            img_file.close();
        }

        // 保存标签为YOLO格式
        // YOLO格式: <class_id> <center_x> <center_y> <width> <height>
        // 对于MNIST，我们将数字放在图像中心，大小为图像的一半
        std::ofstream lbl_file(label_path);
        if (lbl_file.is_open()) {
            lbl_file << static_cast<int>(label) << " 0.5 0.5 0.5 0.5\n";
            lbl_file.close();
        }

        // 显示进度
        if (i % 1000 == 0 || i == images.size() - 1) {
            std::cout << "处理进度: " << i + 1 << "/" << images.size() << " (" 
                      << (i + 1) * 100 / images.size() << "%)" << std::endl;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "用法: " << argv[0] << " <mnist_images_file> <mnist_labels_file> <output_dir>" << std::endl;
        return 1;
    }

    std::string images_file = argv[1];
    std::string labels_file = argv[2];
    std::string output_dir = argv[3];

    // 读取MNIST数据
    std::vector<std::vector<uint8_t>> images;
    std::vector<uint8_t> labels;

    if (!readMNISTImages(images_file, images)) {
        return 1;
    }

    if (!readMNISTLabels(labels_file, labels)) {
        return 1;
    }

    // 检查图像和标签数量是否匹配
    if (images.size() != labels.size()) {
        std::cerr << "错误: 图像数量 (" << images.size() << ") 与标签数量 (" 
                  << labels.size() << ") 不匹配" << std::endl;
        return 1;
    }

    // 转换为YOLO格式
    convertMNISTToYOLO(images, labels, output_dir);

    std::cout << "MNIST数据集转换完成!" << std::endl;
    return 0;
}