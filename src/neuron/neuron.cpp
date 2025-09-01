#include "neuron.h"
#include <random>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace neural_network {
    
Neuron::Neuron(size_t numInputs) 
    : weights_(numInputs), bias_(0.0), activation_type_(ActivationType::SIGMOID), weight_gradients_(numInputs),
      bias_gradient_(0.0), output_(0.0) {
    // 初始化权重和偏置为小的随机数
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-0.5, 0.5);
    
    for (size_t i = 0; i < numInputs; i++) {
        weights_[i] = dis(gen);
    }
    bias_ = dis(gen);
    
    // 初始化默认激活函数
    initializeActivationFunction(activation_type_);
}

double Neuron::forward(const std::vector<double>& inputs) {
    // 计算加权输入和
    double sum = std::inner_product(inputs.begin(), inputs.end(), weights_.begin(), 0.0) + bias_;
    
    // 应用激活函数
    output_ = activation_function_(sum);
    
    return output_;
}

void Neuron::setActivationFunction(ActivationType type) {
    activation_type_ = type;
    initializeActivationFunction(type);
}

void Neuron::setActivationFunction(std::function<double(double)> activation_func) {
    activation_function_ = activation_func;
    activation_type_ = ActivationType::SIGMOID; // 默认设置
}

void Neuron::initializeActivationFunction(ActivationType type) {
    switch (type) {
        case ActivationType::TANH:
            activation_function_ = [](double x) -> double {
                return std::tanh(x);
            };
            break;
            
        case ActivationType::RELU:
            activation_function_ = [](double x) -> double {
                return std::max(0.0, x);
            };
            break;
            
        case ActivationType::SIGMOID:
        default:
            activation_function_ = [](double x) -> double {
                return 1.0 / (1.0 + std::exp(-x));
            };
            break;
    }
}

const std::vector<double>& Neuron::getWeights() const {
    return weights_;
}

void Neuron::setWeights(const std::vector<double>& weights) {
    if (weights.size() == weights_.size()) {
        weights_ = weights;
    }
}

double Neuron::getBias() const {
    return bias_;
}

void Neuron::setBias(double bias) {
    bias_ = bias;
}

const std::vector<double>& Neuron::getWeightGradients() const {
    return weight_gradients_;
}

double Neuron::getBiasGradient() const {
    return bias_gradient_;
}

void Neuron::setGradients(const std::vector<double>& weightGradients, double biasGradient) {
    if (weightGradients.size() == weight_gradients_.size()) {
        weight_gradients_ = weightGradients;
        bias_gradient_ = biasGradient;
    }
}

void Neuron::updateWeights(double learningRate) {
    // 根据梯度更新权重和偏置
    for (size_t i = 0; i < weights_.size(); i++) {
        weights_[i] -= learningRate * weight_gradients_[i];
    }
    bias_ -= learningRate * bias_gradient_;
}

double Neuron::getOutput() const {
    return output_;
}

double Neuron::computeActivationDerivative(double output) const {
    switch (activation_type_) {
        case ActivationType::TANH:
            return 1.0 - output * output;
            
        case ActivationType::RELU:
            return output > 0 ? 1.0 : 0.0;
            
        case ActivationType::SIGMOID:
        default:
            return output * (1.0 - output);
    }
}

} // namespace neural_network