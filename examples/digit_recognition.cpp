#include "../src/network/network.h"
#include "../src/network/layer.h"
#include "../src/neuron/neuron.h"
#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <random>

// 简单的手写数字数据集（3x3像素）
class SimpleDigitDataset {
public:
    // 3x3像素表示简单数字
    static const std::vector<std::vector<double>> digits_0;
    static const std::vector<std::vector<double>> digits_1;
    
    static std::vector<std::pair<std::vector<double>, std::vector<double>>> getTrainingData() {
        std::vector<std::pair<std::vector<double>, std::vector<double>>> data;
        
        // 添加数字0的样本
        for (const auto& sample : digits_0) {
            std::vector<double> target(2, 0.0);
            target[0] = 1.0; // 第一类：数字0
            data.push_back({sample, target});
        }
        
        // 添加数字1的样本
        for (const auto& sample : digits_1) {
            std::vector<double> target(2, 0.0);
            target[1] = 1.0; // 第二类：数字1
            data.push_back({sample, target});
        }
        
        return data;
    }
};

// 数字0的表示（3x3像素）
const std::vector<std::vector<double>> SimpleDigitDataset::digits_0 = {
    {1, 1, 1, 1, 0, 1, 1, 1, 1},  // 0
    {0, 1, 0, 1, 0, 1, 0, 1, 0},  // 0变体
    {1, 1, 1, 1, 0, 1, 1, 1, 1}   // 0变体
};

// 数字1的表示（3x3像素）
const std::vector<std::vector<double>> SimpleDigitDataset::digits_1 = {
    {0, 1, 0, 0, 1, 0, 0, 1, 0},  // 1
    {0, 0, 1, 0, 0, 1, 0, 0, 1},  // 1变体
    {0, 1, 0, 0, 1, 0, 0, 1, 0}   // 1变体
};

int main() {
    std::cout << "手写数字识别示例 - 深度学习神经网络框架" << std::endl;
    
    // 创建神经网络 (9输入 -> 12隐藏 -> 6隐藏 -> 2输出)
    neural_network::Network network;
    
    // 添加隐藏层
    auto hiddenLayer1 = std::make_shared<neural_network::Layer>(12, 9);
    network.addLayer(hiddenLayer1);
    
    auto hiddenLayer2 = std::make_shared<neural_network::Layer>(6, 12);
    network.addLayer(hiddenLayer2);
    
    // 添加输出层
    auto outputLayer = std::make_shared<neural_network::Layer>(2, 6);
    network.addLayer(outputLayer);
    
    // 获取训练数据
    auto trainingData = SimpleDigitDataset::getTrainingData();
    
    std::cout << "训练数据集大小: " << trainingData.size() << std::endl;
    
    std::cout << "训练前的预测结果:" << std::endl;
    for (size_t i = 0; i < trainingData.size(); i++) {
        auto output = network.forward(trainingData[i].first);
        int predicted_class = output[0] > output[1] ? 0 : 1;
        int actual_class = trainingData[i].second[0] > 0.5 ? 0 : 1;
        std::cout << "样本 " << i << ": 实际=" << actual_class 
                  << ", 预测=[" << output[0] << ", " << output[1] << "]"
                  << ", 预测类别=" << predicted_class << std::endl;
    }
    
    // 训练网络
    std::cout << "\n开始训练..." << std::endl;
    const int epochs = 5000;
    const double learningRate = 1.0;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        // 随机打乱数据
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(trainingData.begin(), trainingData.end(), g);
        
        double totalLoss = 0.0;
        for (const auto& sample : trainingData) {
            auto output = network.forward(sample.first);
            totalLoss += network.computeLoss(output, sample.second);
            network.train(sample.first, sample.second, learningRate);
        }
        
        // 每500轮显示一次损失
        if (epoch % 500 == 0) {
            std::cout << "Epoch " << epoch << ", 平均损失: " << totalLoss / trainingData.size() << std::endl;
        }
    }
    
    // 训练后的预测结果
    std::cout << "\n训练后的预测结果:" << std::endl;
    int correct = 0;
    for (size_t i = 0; i < trainingData.size(); i++) {
        auto output = network.forward(trainingData[i].first);
        int predicted_class = output[0] > output[1] ? 0 : 1;
        int actual_class = trainingData[i].second[0] > 0.5 ? 0 : 1;
        if (predicted_class == actual_class) correct++;
        std::cout << "样本 " << i << ": 实际=" << actual_class 
                  << ", 预测=[" << output[0] << ", " << output[1] << "]"
                  << ", 预测类别=" << predicted_class 
                  << (predicted_class == actual_class ? " ✓" : " ✗") << std::endl;
    }
    
    std::cout << "\n准确率: " << (100.0 * correct / trainingData.size()) << "%" << std::endl;
    
    return 0;
}