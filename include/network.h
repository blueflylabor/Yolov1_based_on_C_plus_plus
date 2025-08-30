#ifndef YOLOV1_NETWORK_H
#define YOLOV1_NETWORK_H

#include <vector>
#include <memory>
#include <random>
#include <cmath>

// 激活函数类型
enum class ActivationType {
    NONE,
    RELU,
    LEAKY_RELU,
    SIGMOID
};

// 基础层类
class Layer {
public:
    virtual ~Layer() = default;
    virtual std::vector<float> forward(const std::vector<float>& input) = 0;
    virtual std::vector<float> backward(const std::vector<float>& gradient) = 0;
    virtual void updateWeights(float learning_rate) = 0;
};

// 卷积层
class ConvolutionalLayer : public Layer {
private:
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    ActivationType activation;
    
    std::vector<std::vector<std::vector<std::vector<float>>>> weights; // [out_channels][in_channels][kernel_size][kernel_size]
    std::vector<float> biases; // [out_channels]
    
    // 存储前向传播的输入，用于反向传播
    std::vector<float> last_input;
    std::vector<float> last_output;

public:
    ConvolutionalLayer(int in_channels, int out_channels, int kernel_size, 
                      int stride = 1, int padding = 0, 
                      ActivationType activation = ActivationType::RELU);
    
    std::vector<float> forward(const std::vector<float>& input) override;
    std::vector<float> backward(const std::vector<float>& gradient) override;
    void updateWeights(float learning_rate) override;
    
    // 初始化权重
    void initializeWeights();
};

// 最大池化层
class MaxPoolingLayer : public Layer {
private:
    int kernel_size;
    int stride;
    
    // 存储最大值的索引，用于反向传播
    std::vector<int> max_indices;
    std::vector<float> last_input;

public:
    MaxPoolingLayer(int kernel_size, int stride = 2);
    
    std::vector<float> forward(const std::vector<float>& input) override;
    std::vector<float> backward(const std::vector<float>& gradient) override;
    void updateWeights(float learning_rate) override { /* 池化层没有权重 */ }
};

// 全连接层
class FullyConnectedLayer : public Layer {
private:
    int input_size;
    int output_size;
    ActivationType activation;
    
    std::vector<std::vector<float>> weights; // [output_size][input_size]
    std::vector<float> biases; // [output_size]
    
    std::vector<float> last_input;
    std::vector<float> last_output;

public:
    FullyConnectedLayer(int input_size, int output_size, 
                       ActivationType activation = ActivationType::NONE);
    
    std::vector<float> forward(const std::vector<float>& input) override;
    std::vector<float> backward(const std::vector<float>& gradient) override;
    void updateWeights(float learning_rate) override;
    
    // 初始化权重
    void initializeWeights();
};

// 激活函数层
class ActivationLayer : public Layer {
private:
    ActivationType type;
    std::vector<float> last_input;

public:
    explicit ActivationLayer(ActivationType type);
    
    std::vector<float> forward(const std::vector<float>& input) override;
    std::vector<float> backward(const std::vector<float>& gradient) override;
    void updateWeights(float learning_rate) override { /* 激活层没有权重 */ }
};

// YOLOv1网络
class YOLONetwork {
private:
    std::vector<std::shared_ptr<Layer>> layers;
    int input_width;
    int input_height;
    int input_channels;
    int grid_size;
    int num_boxes;
    int num_classes;

public:
    YOLONetwork(int input_width = 448, int input_height = 448, 
               int input_channels = 3, int grid_size = 7, 
               int num_boxes = 2, int num_classes = 20);
    
    // 构建网络结构
    void buildNetwork();
    
    // 前向传播
    std::vector<float> forward(const std::vector<float>& input);
    
    // 反向传播
    void backward(const std::vector<float>& gradient, float learning_rate);
    
    // 加载预训练权重
    bool loadWeights(const std::string& weight_file);
    
    // 保存权重
    bool saveWeights(const std::string& output_file);
};

#endif // YOLOV1_NETWORK_H