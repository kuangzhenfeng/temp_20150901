#include "network.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <cassert>
#include <sstream>

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
    
    // 从最后一层获取输出
    std::vector<double> outputs;
    auto output_neurons = layers_.back()->getNeurons();
    outputs.reserve(output_neurons.size());
    for (const auto& neuron : output_neurons) {
        outputs.push_back(neuron->getOutput());
    }
    
    // 计算输出层误差
    std::vector<double> output_errors = computeOutputLayerErrors(outputs, targets);
    
    // 从输出层开始反向传播误差
    std::vector<double> errors = output_errors;
    
    // 从最后一层向前遍历
    for (int i = layers_.size() - 1; i >= 0; --i) {
        auto layer = layers_[i];
        auto neurons = layer->getNeurons();
        const std::vector<double>& layer_inputs = layer->getLastInputs();
        
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
            
            // 计算权重梯度
            weight_gradients[j].resize(neuron->getWeights().size());
            for (size_t k = 0; k < weight_gradients[j].size(); k++) {
                weight_gradients[j][k] = error_term * layer_inputs[k];
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

bool Network::saveModel(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    // 写入网络结构信息
    file << layers_.size() << std::endl;
    
    // 写入每层的信息
    for (const auto& layer : layers_) {
        auto neurons = layer->getNeurons();
        file << neurons.size() << " " << neurons[0]->getWeights().size() << std::endl;
        
        // 写入每个神经元的信息
        for (const auto& neuron : neurons) {
            // 写入偏置
            file << neuron->getBias() << std::endl;
            
            // 写入权重
            const auto& weights = neuron->getWeights();
            for (size_t k = 0; k < weights.size(); k++) {
                file << weights[k];
                if (k < weights.size() - 1) {
                    file << " ";
                }
            }
            file << std::endl;
        }
    }
    
    file.close();
    return true;
}

bool Network::loadModel(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    // 读取网络结构信息
    size_t layerCount;
    file >> layerCount;
    
    // 清空现有层
    layers_.clear();
    
    // 读取每层的信息
    for (size_t i = 0; i < layerCount; i++) {
        size_t neuronCount, inputCount;
        file >> neuronCount >> inputCount;
        
        // 创建层
        auto layer = std::make_shared<Layer>(neuronCount, inputCount);
        auto neurons = layer->getNeurons();
        
        // 读取每个神经元的信息
        for (size_t j = 0; j < neuronCount; j++) {
            // 读取偏置
            double bias;
            file >> bias;
            neurons[j]->setBias(bias);
            
            // 读取权重
            std::vector<double> weights(inputCount);
            for (size_t k = 0; k < inputCount; k++) {
                file >> weights[k];
            }
            neurons[j]->setWeights(weights);
        }
        
        layers_.push_back(layer);
    }
    
    file.close();
    return true;
}

} // namespace neural_network