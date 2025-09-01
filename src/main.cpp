#include <iostream>
#include "network/network.h"
#include "network/layer.h"
#include "neuron/neuron.h"
#include <memory>
#include <vector>

int main() {
    std::cout << "深度学习神经网络框架" << std::endl;
    std::cout << "项目初始化成功" << std::endl;
    
    // 创建一个简单的神经网络示例
    neural_network::Network network;
    
    // 添加输入层（2个输入）
    auto inputLayer = std::make_shared<neural_network::Layer>(2, 2);
    network.addLayer(inputLayer);
    
    // 添加隐藏层（3个神经元，每个神经元2个输入）
    auto hiddenLayer = std::make_shared<neural_network::Layer>(3, 2);
    network.addLayer(hiddenLayer);
    
    // 添加输出层（1个神经元，3个输入）
    auto outputLayer = std::make_shared<neural_network::Layer>(1, 3);
    network.addLayer(outputLayer);
    
    std::cout << "已创建示例网络，包含:" << std::endl;
    std::cout << "- 输入层: 2个神经元" << std::endl;
    std::cout << "- 隐藏层: 3个神经元" << std::endl;
    std::cout << "- 输出层: 1个神经元" << std::endl;
    
    return 0;
}