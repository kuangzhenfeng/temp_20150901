#include "neuron.h"
#include <numeric>
#include <cmath>

namespace neural_network {
    
Neuron::Neuron(int id) 
    : id_(id), membrane_potential_(0.0), threshold_(1.0) {
    // 默认使用sigmoid激活函数
    activation_function_ = [](double x) -> double {
        return 1.0 / (1.0 + std::exp(-x));
    };
}

int Neuron::getId() const {
    return id_;
}

void Neuron::addInputSignal(double signal) {
    input_signals_.push_back(signal);
}

double Neuron::computeOutput() {
    // 计算所有输入信号的总和
    double sum = std::accumulate(input_signals_.begin(), input_signals_.end(), 0.0);
    
    // 更新膜电位
    membrane_potential_ = sum;
    
    // 应用激活函数
    return activation_function_(membrane_potential_);
}

void Neuron::clearInputSignals() {
    input_signals_.clear();
}

void Neuron::setActivationFunction(std::function<double(double)> activation_func) {
    activation_function_ = activation_func;
}

double Neuron::getMembranePotential() const {
    return membrane_potential_;
}

void Neuron::setMembranePotential(double potential) {
    membrane_potential_ = potential;
}

} // namespace neural_network