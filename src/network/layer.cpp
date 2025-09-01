#include "layer.h"
#include "../neuron/neuron.h"
#include <algorithm>

namespace neural_network {
    
Layer::Layer(size_t numNeurons, size_t numInputs) 
    : last_inputs_(numInputs), last_outputs_(numNeurons) {
    // 创建指定数量的神经元
    for (size_t i = 0; i < numNeurons; i++) {
        neurons_.push_back(std::make_shared<Neuron>(numInputs));
    }
}

std::vector<double> Layer::forward(const std::vector<double>& inputs) {
    // 存储输入和输出用于反向传播
    last_inputs_ = inputs;
    
    std::vector<double> outputs;
    outputs.reserve(neurons_.size());
    
    // 对每个神经元执行前向传播
    for (auto& neuron : neurons_) {
        outputs.push_back(neuron->forward(inputs));
    }
    
    last_outputs_ = outputs;
    return outputs;
}

const std::vector<std::shared_ptr<Neuron>>& Layer::getNeurons() const {
    return neurons_;
}

size_t Layer::size() const {
    return neurons_.size();
}

void Layer::setGradients(const std::vector<std::vector<double>>& weightGradients, 
                         const std::vector<double>& biasGradients) {
    if (weightGradients.size() == neurons_.size() && biasGradients.size() == neurons_.size()) {
        for (size_t i = 0; i < neurons_.size(); i++) {
            neurons_[i]->setGradients(weightGradients[i], biasGradients[i]);
        }
    }
}

void Layer::updateWeights(double learningRate) {
    for (auto& neuron : neurons_) {
        neuron->updateWeights(learningRate);
    }
}

const std::vector<double>& Layer::getLastInputs() const {
    return last_inputs_;
}

const std::vector<double>& Layer::getLastOutputs() const {
    return last_outputs_;
}

} // namespace neural_network