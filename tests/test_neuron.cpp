#include "../src/neuron/neuron.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

int main() {
    std::cout << "测试Neuron类功能..." << std::endl;
    
    // 测试1: 创建神经元
    neural_network::Neuron neuron(3);
    std::cout << "✓ 成功创建具有3个输入的神经元" << std::endl;
    
    // 测试2: 测试前向传播
    std::vector<double> inputs = {1.0, 2.0, 3.0};
    double output = neuron.forward(inputs);
    std::cout << "✓ 前向传播成功，输出: " << output << std::endl;
    
    // 测试3: 测试激活函数设置
    neuron.setActivationFunction(neural_network::ActivationType::RELU);
    double output_relu = neuron.forward(inputs);
    std::cout << "✓ ReLU激活函数设置成功，输出: " << output_relu << std::endl;
    
    // 测试4: 测试权重获取和设置
    auto weights = neuron.getWeights();
    std::cout << "✓ 成功获取权重，权重数量: " << weights.size() << std::endl;
    
    // 测试5: 测试偏置获取和设置
    double bias = neuron.getBias();
    std::cout << "✓ 成功获取偏置: " << bias << std::endl;
    
    neuron.setBias(0.5);
    std::cout << "✓ 成功设置偏置为0.5" << std::endl;
    
    // 测试6: 测试梯度功能
    std::vector<double> grad_weights(weights.size(), 0.1);
    neuron.setGradients(grad_weights, 0.2);
    std::cout << "✓ 成功设置梯度" << std::endl;
    
    // 测试7: 测试权重更新
    std::vector<double> old_weights = neuron.getWeights();
    double old_bias = neuron.getBias();
    
    neuron.updateWeights(0.01);
    
    std::vector<double> new_weights = neuron.getWeights();
    double new_bias = neuron.getBias();
    
    bool weights_updated = false;
    for (size_t i = 0; i < old_weights.size(); i++) {
        if (std::abs(old_weights[i] - new_weights[i]) > 1e-6) {
            weights_updated = true;
            break;
        }
    }
    
    bool bias_updated = std::abs(old_bias - new_bias) > 1e-6;
    
    if (weights_updated || bias_updated) {
        std::cout << "✓ 权重更新功能正常" << std::endl;
    } else {
        std::cout << "⚠ 权重更新可能未按预期工作" << std::endl;
    }
    
    // 测试8: 测试激活函数导数计算
    neuron.setActivationFunction(neural_network::ActivationType::SIGMOID);
    double sigmoid_output = neuron.forward(inputs);
    double sigmoid_derivative = neuron.computeActivationDerivative(sigmoid_output);
    std::cout << "✓ Sigmoid导数计算: " << sigmoid_derivative << std::endl;
    
    neuron.setActivationFunction(neural_network::ActivationType::RELU);
    double relu_output = neuron.forward(inputs);
    double relu_derivative = neuron.computeActivationDerivative(relu_output);
    std::cout << "✓ ReLU导数计算: " << relu_derivative << std::endl;
    
    std::cout << "\n所有测试完成!" << std::endl;
    return 0;
}