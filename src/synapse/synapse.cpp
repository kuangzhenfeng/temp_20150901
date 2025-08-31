#include "synapse.h"
#include "../neuron/neuron.h"

namespace neural_network {
    
Synapse::Synapse(std::shared_ptr<Neuron> pre, std::shared_ptr<Neuron> post, double weight)
    : pre_neuron_(pre), post_neuron_(post), weight_(weight) {
}

std::shared_ptr<Neuron> Synapse::getPreNeuron() const {
    return pre_neuron_;
}

std::shared_ptr<Neuron> Synapse::getPostNeuron() const {
    return post_neuron_;
}

double Synapse::getWeight() const {
    return weight_;
}

void Synapse::setWeight(double weight) {
    weight_ = weight;
}

void Synapse::transmit() {
    // 这里应该在神经网络更新循环中实现信号传递逻辑
    // 当前实现仅作示意
}

} // namespace neural_network