#include "../src/network/network.h"
#include "../src/network/layer.h"
#include "../src/neuron/neuron.h"
#include <iostream>
#include <vector>
#include <memory>
#include <cmath>

int main() {
    std::cout << "测试Network类功能..." << std::endl;
    
    // 测试1: 创建网络
    neural_network::Network network;
    std::cout << "✓ 成功创建网络" << std::endl;
    
    // 测试2: 添加网络层
    auto hiddenLayer = std::make_shared<neural_network::Layer>(3, 2);
    network.addLayer(hiddenLayer);
    
    auto outputLayer = std::make_shared<neural_network::Layer>(1, 3);
    network.addLayer(outputLayer);
    
    std::cout << "✓ 成功添加网络层，网络层数: " << network.getLayerCount() << std::endl;
    
    // 测试3: 前向传播
    std::vector<double> inputs = {0.5, 0.8};
    std::vector<double> outputs = network.forward(inputs);
    
    std::cout << "✓ 前向传播成功，输出维度: " << outputs.size() << std::endl;
    std::cout << "  输出值: " << outputs[0] << std::endl;
    
    // 测试4: 损失计算
    std::vector<double> targets = {1.0};
    double loss = network.computeLoss(outputs, targets);
    std::cout << "✓ 损失计算成功，损失值: " << loss << std::endl;
    
    // 测试5: 训练功能
    network.train(inputs, targets, 0.01);
    std::cout << "✓ 训练功能调用成功" << std::endl;
    
    // 测试6: 不同损失函数
    network.setLossFunctionType(neural_network::LossFunctionType::CROSS_ENTROPY);
    double cross_entropy_loss = network.computeLoss(outputs, targets);
    std::cout << "✓ 交叉熵损失计算成功，损失值: " << cross_entropy_loss << std::endl;
    
    // 测试7: 获取层功能
    auto layer0 = network.getLayer(0);
    auto layer1 = network.getLayer(1);
    auto layer2 = network.getLayer(2); // 不存在的层
    
    if (layer0 && layer1 && !layer2) {
        std::cout << "✓ 网络层获取功能正常" << std::endl;
    } else {
        std::cout << "⚠ 网络层获取功能可能存在问题" << std::endl;
    }
    
    std::cout << "\n所有网络测试完成!" << std::endl;
    return 0;
}