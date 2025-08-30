#include "../include/network.h"
#include <fstream>
#include <iostream>

// 卷积层实现
ConvolutionalLayer::ConvolutionalLayer(int in_channels, int out_channels, int kernel_size, 
                                     int stride, int padding, ActivationType activation)
    : in_channels(in_channels), out_channels(out_channels), kernel_size(kernel_size),
      stride(stride), padding(padding), activation(activation) {
    
    // 初始化权重和偏置
    weights.resize(out_channels, std::vector<std::vector<std::vector<float>>>
                  (in_channels, std::vector<std::vector<float>>
                  (kernel_size, std::vector<float>(kernel_size, 0.0f))));
    biases.resize(out_channels, 0.0f);
    
    initializeWeights();
}

void ConvolutionalLayer::initializeWeights() {
    // 使用He初始化
    std::random_device rd;
    std::mt19937 gen(rd());
    float scale = std::sqrt(2.0f / (in_channels * kernel_size * kernel_size));
    std::normal_distribution<float> distribution(0.0f, scale);
    
    for (int oc = 0; oc < out_channels; oc++) {
        for (int ic = 0; ic < in_channels; ic++) {
            for (int i = 0; i < kernel_size; i++) {
                for (int j = 0; j < kernel_size; j++) {
                    weights[oc][ic][i][j] = distribution(gen);
                }
            }
        }
        biases[oc] = 0.0f; // 初始化偏置为0
    }
}

std::vector<float> ConvolutionalLayer::forward(const std::vector<float>& input) {
    // 保存输入用于反向传播
    last_input = input;
    
    // 假设输入是一个3D张量，展平为1D向量
    // 输入尺寸: [in_channels, height, width]
    int height = std::sqrt(input.size() / in_channels);
    int width = height;
    
    // 计算输出尺寸
    int out_height = (height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (width + 2 * padding - kernel_size) / stride + 1;
    
    // 初始化输出
    std::vector<float> output(out_channels * out_height * out_width, 0.0f);
    
    // 执行卷积操作
    for (int oc = 0; oc < out_channels; oc++) {
        for (int oh = 0; oh < out_height; oh++) {
            for (int ow = 0; ow < out_width; ow++) {
                float sum = biases[oc];
                
                for (int ic = 0; ic < in_channels; ic++) {
                    for (int kh = 0; kh < kernel_size; kh++) {
                        for (int kw = 0; kw < kernel_size; kw++) {
                            int ih = oh * stride + kh - padding;
                            int iw = ow * stride + kw - padding;
                            
                            // 检查边界
                            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                int input_idx = ic * height * width + ih * width + iw;
                                sum += input[input_idx] * weights[oc][ic][kh][kw];
                            }
                        }
                    }
                }
                
                // 应用激活函数
                float activated_value = sum;
                if (activation == ActivationType::RELU) {
                    activated_value = std::max(0.0f, sum);
                } else if (activation == ActivationType::LEAKY_RELU) {
                    activated_value = sum > 0 ? sum : 0.1f * sum;
                } else if (activation == ActivationType::SIGMOID) {
                    activated_value = 1.0f / (1.0f + std::exp(-sum));
                }
                
                int output_idx = oc * out_height * out_width + oh * out_width + ow;
                output[output_idx] = activated_value;
            }
        }
    }
    
    last_output = output;
    return output;
}

std::vector<float> ConvolutionalLayer::backward(const std::vector<float>& gradient) {
    // 这里简化了反向传播的实现
    // 在实际应用中，需要计算权重和偏置的梯度，并更新它们
    // 这里只返回输入梯度
    
    // 假设梯度是一个3D张量，展平为1D向量
    // 梯度尺寸: [out_channels, out_height, out_width]
    int out_height = std::sqrt(gradient.size() / out_channels);
    int out_width = out_height;
    
    // 输入尺寸
    int height = std::sqrt(last_input.size() / in_channels);
    int width = height;
    
    // 初始化输入梯度
    std::vector<float> input_gradient(in_channels * height * width, 0.0f);
    
    // 计算输入梯度
    for (int oc = 0; oc < out_channels; oc++) {
        for (int oh = 0; oh < out_height; oh++) {
            for (int ow = 0; ow < out_width; ow++) {
                int output_idx = oc * out_height * out_width + oh * out_width + ow;
                float output_grad = gradient[output_idx];
                
                // 应用激活函数的梯度
                float activated_grad = output_grad;
                if (activation == ActivationType::RELU) {
                    activated_grad = last_output[output_idx] > 0 ? output_grad : 0.0f;
                } else if (activation == ActivationType::LEAKY_RELU) {
                    activated_grad = last_output[output_idx] > 0 ? output_grad : 0.1f * output_grad;
                } else if (activation == ActivationType::SIGMOID) {
                    float sigmoid_output = last_output[output_idx];
                    activated_grad = output_grad * sigmoid_output * (1.0f - sigmoid_output);
                }
                
                for (int ic = 0; ic < in_channels; ic++) {
                    for (int kh = 0; kh < kernel_size; kh++) {
                        for (int kw = 0; kw < kernel_size; kw++) {
                            int ih = oh * stride + kh - padding;
                            int iw = ow * stride + kw - padding;
                            
                            // 检查边界
                            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                int input_idx = ic * height * width + ih * width + iw;
                                input_gradient[input_idx] += activated_grad * weights[oc][ic][kh][kw];
                            }
                        }
                    }
                }
            }
        }
    }
    
    return input_gradient;
}

void ConvolutionalLayer::updateWeights(float learning_rate) {
    // 简化的权重更新实现
    // 在实际应用中，需要使用计算得到的梯度来更新权重和偏置
}

// 最大池化层实现
MaxPoolingLayer::MaxPoolingLayer(int kernel_size, int stride)
    : kernel_size(kernel_size), stride(stride) {}

std::vector<float> MaxPoolingLayer::forward(const std::vector<float>& input) {
    // 保存输入用于反向传播
    last_input = input;
    
    // 假设输入是一个3D张量，展平为1D向量
    // 输入尺寸: [channels, height, width]
    int channels = 1; // 简化处理，假设只有一个通道
    int height = std::sqrt(input.size());
    int width = height;
    
    // 计算输出尺寸
    int out_height = (height - kernel_size) / stride + 1;
    int out_width = (width - kernel_size) / stride + 1;
    
    // 初始化输出和最大值索引
    std::vector<float> output(channels * out_height * out_width, 0.0f);
    max_indices.resize(channels * out_height * out_width, -1);
    
    // 执行最大池化操作
    for (int c = 0; c < channels; c++) {
        for (int oh = 0; oh < out_height; oh++) {
            for (int ow = 0; ow < out_width; ow++) {
                float max_val = -std::numeric_limits<float>::max();
                int max_idx = -1;
                
                for (int kh = 0; kh < kernel_size; kh++) {
                    for (int kw = 0; kw < kernel_size; kw++) {
                        int ih = oh * stride + kh;
                        int iw = ow * stride + kw;
                        
                        int input_idx = c * height * width + ih * width + iw;
                        if (input[input_idx] > max_val) {
                            max_val = input[input_idx];
                            max_idx = input_idx;
                        }
                    }
                }
                
                int output_idx = c * out_height * out_width + oh * out_width + ow;
                output[output_idx] = max_val;
                max_indices[output_idx] = max_idx;
            }
        }
    }
    
    return output;
}

std::vector<float> MaxPoolingLayer::backward(const std::vector<float>& gradient) {
    // 初始化输入梯度
    std::vector<float> input_gradient(last_input.size(), 0.0f);
    
    // 反向传播梯度
    for (size_t i = 0; i < gradient.size(); i++) {
        if (max_indices[i] >= 0) {
            input_gradient[max_indices[i]] += gradient[i];
        }
    }
    
    return input_gradient;
}

// 全连接层实现
FullyConnectedLayer::FullyConnectedLayer(int input_size, int output_size, ActivationType activation)
    : input_size(input_size), output_size(output_size), activation(activation) {
    
    // 初始化权重和偏置
    weights.resize(output_size, std::vector<float>(input_size, 0.0f));
    biases.resize(output_size, 0.0f);
    
    initializeWeights();
}

void FullyConnectedLayer::initializeWeights() {
    // 使用Xavier初始化
    std::random_device rd;
    std::mt19937 gen(rd());
    float scale = std::sqrt(6.0f / (input_size + output_size));
    std::uniform_real_distribution<float> distribution(-scale, scale);
    
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < input_size; j++) {
            weights[i][j] = distribution(gen);
        }
        biases[i] = 0.0f; // 初始化偏置为0
    }
}

std::vector<float> FullyConnectedLayer::forward(const std::vector<float>& input) {
    // 保存输入用于反向传播
    last_input = input;
    
    // 初始化输出
    std::vector<float> output(output_size, 0.0f);
    
    // 执行全连接操作
    for (int i = 0; i < output_size; i++) {
        float sum = biases[i];
        for (int j = 0; j < input_size; j++) {
            sum += input[j] * weights[i][j];
        }
        
        // 应用激活函数
        float activated_value = sum;
        if (activation == ActivationType::RELU) {
            activated_value = std::max(0.0f, sum);
        } else if (activation == ActivationType::LEAKY_RELU) {
            activated_value = sum > 0 ? sum : 0.1f * sum;
        } else if (activation == ActivationType::SIGMOID) {
            activated_value = 1.0f / (1.0f + std::exp(-sum));
        }
        
        output[i] = activated_value;
    }
    
    last_output = output;
    return output;
}

std::vector<float> FullyConnectedLayer::backward(const std::vector<float>& gradient) {
    // 初始化输入梯度
    std::vector<float> input_gradient(input_size, 0.0f);
    
    // 计算输入梯度
    for (int i = 0; i < output_size; i++) {
        // 应用激活函数的梯度
        float activated_grad = gradient[i];
        if (activation == ActivationType::RELU) {
            activated_grad = last_output[i] > 0 ? gradient[i] : 0.0f;
        } else if (activation == ActivationType::LEAKY_RELU) {
            activated_grad = last_output[i] > 0 ? gradient[i] : 0.1f * gradient[i];
        } else if (activation == ActivationType::SIGMOID) {
            float sigmoid_output = last_output[i];
            activated_grad = gradient[i] * sigmoid_output * (1.0f - sigmoid_output);
        }
        
        for (int j = 0; j < input_size; j++) {
            input_gradient[j] += activated_grad * weights[i][j];
        }
    }
    
    return input_gradient;
}

void FullyConnectedLayer::updateWeights(float learning_rate) {
    // 简化的权重更新实现
    // 在实际应用中，需要使用计算得到的梯度来更新权重和偏置
}

// 激活函数层实现
ActivationLayer::ActivationLayer(ActivationType type) : type(type) {}

std::vector<float> ActivationLayer::forward(const std::vector<float>& input) {
    // 保存输入用于反向传播
    last_input = input;
    
    // 初始化输出
    std::vector<float> output(input.size(), 0.0f);
    
    // 应用激活函数
    for (size_t i = 0; i < input.size(); i++) {
        if (type == ActivationType::RELU) {
            output[i] = std::max(0.0f, input[i]);
        } else if (type == ActivationType::LEAKY_RELU) {
            output[i] = input[i] > 0 ? input[i] : 0.1f * input[i];
        } else if (type == ActivationType::SIGMOID) {
            output[i] = 1.0f / (1.0f + std::exp(-input[i]));
        } else {
            output[i] = input[i]; // 无激活函数
        }
    }
    
    return output;
}

std::vector<float> ActivationLayer::backward(const std::vector<float>& gradient) {
    // 初始化输入梯度
    std::vector<float> input_gradient(gradient.size(), 0.0f);
    
    // 计算输入梯度
    for (size_t i = 0; i < gradient.size(); i++) {
        if (type == ActivationType::RELU) {
            input_gradient[i] = last_input[i] > 0 ? gradient[i] : 0.0f;
        } else if (type == ActivationType::LEAKY_RELU) {
            input_gradient[i] = last_input[i] > 0 ? gradient[i] : 0.1f * gradient[i];
        } else if (type == ActivationType::SIGMOID) {
            float sigmoid_output = 1.0f / (1.0f + std::exp(-last_input[i]));
            input_gradient[i] = gradient[i] * sigmoid_output * (1.0f - sigmoid_output);
        } else {
            input_gradient[i] = gradient[i]; // 无激活函数
        }
    }
    
    return input_gradient;
}

// YOLONetwork实现
YOLONetwork::YOLONetwork(int input_width, int input_height, int input_channels,
                       int grid_size, int num_boxes, int num_classes)
    : input_width(input_width), input_height(input_height), input_channels(input_channels),
      grid_size(grid_size), num_boxes(num_boxes), num_classes(num_classes) {
    
    buildNetwork();
}

void YOLONetwork::buildNetwork() {
    // 构建YOLOv1网络结构
    // 参考论文中的网络架构
    
    // 第1层: 卷积 + 最大池化
    layers.push_back(std::make_shared<ConvolutionalLayer>(input_channels, 64, 7, 2, 3, ActivationType::LEAKY_RELU));
    layers.push_back(std::make_shared<MaxPoolingLayer>(2, 2));
    
    // 第2层: 卷积 + 最大池化
    layers.push_back(std::make_shared<ConvolutionalLayer>(64, 192, 3, 1, 1, ActivationType::LEAKY_RELU));
    layers.push_back(std::make_shared<MaxPoolingLayer>(2, 2));
    
    // 第3-5层: 卷积
    layers.push_back(std::make_shared<ConvolutionalLayer>(192, 128, 1, 1, 0, ActivationType::LEAKY_RELU));
    layers.push_back(std::make_shared<ConvolutionalLayer>(128, 256, 3, 1, 1, ActivationType::LEAKY_RELU));
    layers.push_back(std::make_shared<ConvolutionalLayer>(256, 256, 1, 1, 0, ActivationType::LEAKY_RELU));
    layers.push_back(std::make_shared<ConvolutionalLayer>(256, 512, 3, 1, 1, ActivationType::LEAKY_RELU));
    layers.push_back(std::make_shared<MaxPoolingLayer>(2, 2));
    
    // 第6-13层: 卷积
    for (int i = 0; i < 4; i++) {
        layers.push_back(std::make_shared<ConvolutionalLayer>(512, 256, 1, 1, 0, ActivationType::LEAKY_RELU));
        layers.push_back(std::make_shared<ConvolutionalLayer>(256, 512, 3, 1, 1, ActivationType::LEAKY_RELU));
    }
    layers.push_back(std::make_shared<ConvolutionalLayer>(512, 512, 1, 1, 0, ActivationType::LEAKY_RELU));
    layers.push_back(std::make_shared<ConvolutionalLayer>(512, 1024, 3, 1, 1, ActivationType::LEAKY_RELU));
    layers.push_back(std::make_shared<MaxPoolingLayer>(2, 2));
    
    // 第14-20层: 卷积
    for (int i = 0; i < 2; i++) {
        layers.push_back(std::make_shared<ConvolutionalLayer>(1024, 512, 1, 1, 0, ActivationType::LEAKY_RELU));
        layers.push_back(std::make_shared<ConvolutionalLayer>(512, 1024, 3, 1, 1, ActivationType::LEAKY_RELU));
    }
    layers.push_back(std::make_shared<ConvolutionalLayer>(1024, 1024, 3, 1, 1, ActivationType::LEAKY_RELU));
    layers.push_back(std::make_shared<ConvolutionalLayer>(1024, 1024, 3, 2, 1, ActivationType::LEAKY_RELU));
    
    // 第21-24层: 卷积
    layers.push_back(std::make_shared<ConvolutionalLayer>(1024, 1024, 3, 1, 1, ActivationType::LEAKY_RELU));
    layers.push_back(std::make_shared<ConvolutionalLayer>(1024, 1024, 3, 1, 1, ActivationType::LEAKY_RELU));
    
    // 全连接层
    int fc_input_size = 1024 * 7 * 7; // 假设最后的特征图大小是7x7
    layers.push_back(std::make_shared<FullyConnectedLayer>(fc_input_size, 4096, ActivationType::LEAKY_RELU));
    layers.push_back(std::make_shared<FullyConnectedLayer>(4096, grid_size * grid_size * (num_boxes * 5 + num_classes), ActivationType::NONE));
}

std::vector<float> YOLONetwork::forward(const std::vector<float>& input) {
    std::vector<float> output = input;
    
    // 前向传播
    for (const auto& layer : layers) {
        output = layer->forward(output);
    }
    
    return output;
}

void YOLONetwork::backward(const std::vector<float>& gradient, float learning_rate) {
    std::vector<float> current_gradient = gradient;
    
    // 反向传播
    for (int i = layers.size() - 1; i >= 0; i--) {
        current_gradient = layers[i]->backward(current_gradient);
        layers[i]->updateWeights(learning_rate);
    }
}

bool YOLONetwork::loadWeights(const std::string& weight_file) {
    std::ifstream file(weight_file, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open weight file: " << weight_file << std::endl;
        return false;
    }
    
    // 加载权重的实现
    // 这里简化了实现，实际应用中需要根据权重文件的格式进行解析
    
    file.close();
    return true;
}

bool YOLONetwork::saveWeights(const std::string& output_file) {
    std::ofstream file(output_file, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open output file: " << output_file << std::endl;
        return false;
    }
    
    // 保存权重的实现
    // 这里简化了实现，实际应用中需要根据权重文件的格式进行保存
    
    file.close();
    return true;
}