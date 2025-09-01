#include "network.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <cassert>

namespace neural_network {

Network::Network() : loss_function_type_(LossFunctionType::MEAN_SQUARED_ERROR) {}

Network::~Network() = default;

void Network::addLayer(std::shared_ptr<Layer> layer) {
    layers_.push_back(layer);
}

std::vector<double> Network::forward(const std::vector<double>& inputs) {
    std::vector<double> outputs = inputs;
    
    // 逐层进行前向传播
    for (auto& layer : layers_) {
        outputs = layer->forward(outputs);
    }
    
    return outputs;
}

void Network::train(const std::vector<double>& inputs, const std::vector<double>& targets, double learningRate) {
    // 前向传播
    std::vector<double> outputs = forward(inputs);
    
    // 反向传播
    backpropagate(targets, learningRate);
}

void Network::backpropagate(const std::vector<double>& targets, double learningRate) {
    if (layers_.empty()) return;
    
    // 存储每层的输出值，用于反向传播计算
    std::vector<std::vector<double>> layer_outputs(layers_.size());
    std::vector<std::vector<double>> layer_inputs(layers_.size());
    
    // 获取每层的输入和输出
    // 注意：这需要修改Layer类以存储最近的输入和输出
    for (size_t i = 0; i < layers_.size(); ++i) {
        // 这里简化处理，实际实现需要Layer类支持获取最近的输入和输出
    }
    
    // 计算输出层误差
    std::vector<double> outputs;
    auto output_neurons = layers_.back()->getNeurons();
    outputs.reserve(output_neurons.size());
    for (const auto& neuron : output_neurons) {
        outputs.push_back(neuron->getOutput());
    }
    
    std::vector<double> output_errors = computeOutputLayerErrors(outputs, targets);
    
    // 从输出层开始反向传播误差
    std::vector<double> errors = output_errors;
    
    // 从最后一层向前遍历
    for (int i = layers_.size() - 1; i >= 0; --i) {
        auto layer = layers_[i];
        auto neurons = layer->getNeurons();
        
        // 计算当前层的梯度
        std::vector<std::vector<double>> weight_gradients(neurons.size());
        std::vector<double> bias_gradients(neurons.size());
        
        std::vector<double> new_errors;
        if (i > 0) { // 非输入层
            new_errors.resize(layers_[i-1]->size(), 0.0);
        }
        
        for (size_t j = 0; j < neurons.size(); j++) {
            auto neuron = neurons[j];
            double output = neuron->getOutput();
            double derivative = neuron->computeActivationDerivative(output);
            
            // 计算梯度
            double error_term = errors[j] * derivative;
            bias_gradients[j] = error_term;
            
            // 注意：这里需要知道进入该神经元的输入值
            // 为简化，我们假设有一个方法可以获取这些输入
            weight_gradients[j].resize(neuron->getWeights().size());
            // 简化的权重梯度计算
            for (size_t k = 0; k < weight_gradients[j].size(); k++) {
                weight_gradients[j][k] = error_term; // 简化处理
            }
            
            // 传播误差到前一层
            if (i > 0) {
                const auto& weights = neuron->getWeights();
                for (size_t k = 0; k < weights.size() && k < new_errors.size(); k++) {
                    new_errors[k] += errors[j] * weights[k];
                }
            }
        }
        
        // 设置梯度并更新权重
        layer->setGradients(weight_gradients, bias_gradients);
        layer->updateWeights(learningRate);
        
        errors = new_errors;
    }
}

std::vector<double> Network::computeOutputLayerErrors(const std::vector<double>& outputs, 
                                                     const std::vector<double>& targets) const {
    assert(outputs.size() == targets.size());
    
    std::vector<double> errors(outputs.size());
    
    switch (loss_function_type_) {
        case LossFunctionType::CROSS_ENTROPY:
            // 交叉熵损失函数的误差就是简单的差值
            for (size_t i = 0; i < outputs.size(); i++) {
                errors[i] = outputs[i] - targets[i];
            }
            break;
            
        case LossFunctionType::MEAN_SQUARED_ERROR:
        default:
            // 均方误差损失函数的误差
            for (size_t i = 0; i < outputs.size(); i++) {
                double error = outputs[i] - targets[i];
                // 乘以激活函数的导数
                auto output_neuron = layers_.back()->getNeurons()[i];
                double derivative = output_neuron->computeActivationDerivative(outputs[i]);
                errors[i] = error * derivative;
            }
            break;
    }
    
    return errors;
}

double Network::computeLoss(const std::vector<double>& outputs, const std::vector<double>& targets) const {
    if (outputs.size() != targets.size()) {
        return 0.0;
    }
    
    double loss = 0.0;
    
    switch (loss_function_type_) {
        case LossFunctionType::CROSS_ENTROPY:
            for (size_t i = 0; i < outputs.size(); i++) {
                // 避免log(0)
                double clipped_output = std::max(1e-15, std::min(1.0 - 1e-15, outputs[i]));
                loss -= targets[i] * std::log(clipped_output) + (1 - targets[i]) * std::log(1 - clipped_output);
            }
            loss /= outputs.size();
            break;
            
        case LossFunctionType::MEAN_SQUARED_ERROR:
        default:
            // 计算均方误差
            for (size_t i = 0; i < outputs.size(); i++) {
                double error = outputs[i] - targets[i];
                loss += error * error;
            }
            loss /= outputs.size();
            break;
    }
    
    return loss;
}

size_t Network::getLayerCount() const {
    return layers_.size();
}

std::shared_ptr<Layer> Network::getLayer(size_t index) const {
    if (index < layers_.size()) {
        return layers_[index];
    }
    return nullptr;
}

void Network::setLossFunctionType(LossFunctionType type) {
    loss_function_type_ = type;
}

} // namespace neural_network