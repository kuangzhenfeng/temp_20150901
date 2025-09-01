#include "../src/network/network.h"
#include "../src/network/layer.h"
#include "../src/neuron/neuron.h"
#include <iostream>
#include <vector>
#include <memory>
#include <cmath>

int main() {
    std::cout << "XOR问题示例 - 深度学习神经网络框架" << std::endl;
    
    // 创建神经网络 (2输入 -> 4隐藏 -> 1输出)
    neural_network::Network network;
    
    // 添加隐藏层 (4个神经元，每个有2个输入)
    auto hiddenLayer = std::make_shared<neural_network::Layer>(4, 2);
    network.addLayer(hiddenLayer);
    
    // 添加输出层 (1个神经元，有4个输入)
    auto outputLayer = std::make_shared<neural_network::Layer>(1, 4);
    network.addLayer(outputLayer);
    
    // XOR训练数据
    std::vector<std::vector<double>> inputs = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    
    std::vector<std::vector<double>> targets = {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };
    
    std::cout << "训练前的输出:" << std::endl;
    for (size_t i = 0; i < inputs.size(); i++) {
        auto output = network.forward(inputs[i]);
        std::cout << "输入: [" << inputs[i][0] << ", " << inputs[i][1] << "] "
                  << "目标: " << targets[i][0] << " "
                  << "输出: " << output[0] << std::endl;
    }
    
    // 训练网络
    std::cout << "\n开始训练..." << std::endl;
    const int epochs = 10000;
    const double learningRate = 1.0;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (size_t i = 0; i < inputs.size(); i++) {
            network.train(inputs[i], targets[i], learningRate);
        }
        
        // 每1000轮显示一次损失
        if (epoch % 1000 == 0) {
            double totalLoss = 0.0;
            for (size_t i = 0; i < inputs.size(); i++) {
                auto output = network.forward(inputs[i]);
                totalLoss += network.computeLoss(output, targets[i]);
            }
            std::cout << "Epoch " << epoch << ", Loss: " << totalLoss / inputs.size() << std::endl;
        }
    }
    
    // 训练后的输出
    std::cout << "\n训练后的输出:" << std::endl;
    for (size_t i = 0; i < inputs.size(); i++) {
        auto output = network.forward(inputs[i]);
        std::cout << "输入: [" << inputs[i][0] << ", " << inputs[i][1] << "] "
                  << "目标: " << targets[i][0] << " "
                  << "输出: " << output[0] << std::endl;
    }
    
    // 保存模型
    std::cout << "\n保存模型到文件..." << std::endl;
    if (network.saveModel("xor_model.dat")) {
        std::cout << "模型保存成功!" << std::endl;
    } else {
        std::cout << "模型保存失败!" << std::endl;
    }
    
    // 创建一个新的网络并加载模型
    std::cout << "\n创建新网络并加载模型..." << std::endl;
    neural_network::Network loadedNetwork;
    
    // 添加相同结构的层
    auto loadedHiddenLayer = std::make_shared<neural_network::Layer>(4, 2);
    loadedNetwork.addLayer(loadedHiddenLayer);
    
    auto loadedOutputLayer = std::make_shared<neural_network::Layer>(1, 4);
    loadedNetwork.addLayer(loadedOutputLayer);
    
    // 加载模型
    if (loadedNetwork.loadModel("xor_model.dat")) {
        std::cout << "模型加载成功!" << std::endl;
        
        // 测试加载的模型
        std::cout << "\n加载模型后的输出:" << std::endl;
        for (size_t i = 0; i < inputs.size(); i++) {
            auto output = loadedNetwork.forward(inputs[i]);
            std::cout << "输入: [" << inputs[i][0] << ", " << inputs[i][1] << "] "
                      << "目标: " << targets[i][0] << " "
                      << "输出: " << output[0] << std::endl;
        }
    } else {
        std::cout << "模型加载失败!" << std::endl;
    }
    
    return 0;
}