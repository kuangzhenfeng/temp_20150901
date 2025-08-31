#include "network.h"
#include <algorithm>
#include "../neuron/neuron.h"
#include "../synapse/synapse.h"

namespace neural_network {

Network::Network() = default;

Network::~Network() = default;

void Network::addNeuron(std::shared_ptr<Neuron> neuron) {
    neurons_.push_back(neuron);
}

void Network::removeNeuron(std::shared_ptr<Neuron> neuron) {
    neurons_.erase(std::remove(neurons_.begin(), neurons_.end(), neuron), neurons_.end());
}

void Network::addSynapse(std::shared_ptr<Synapse> synapse) {
    synapses_.push_back(synapse);
}

void Network::removeSynapse(std::shared_ptr<Synapse> synapse) {
    synapses_.erase(std::remove(synapses_.begin(), synapses_.end(), synapse), synapses_.end());
}

void Network::update() {
    // 先计算所有神经元的输出
    for (auto& neuron : neurons_) {
        neuron->computeOutput();
    }

    // 然后通过突触传递信号
    for (auto& synapse : synapses_) {
        // 这里简化处理，实际应该考虑神经元的输入和输出机制
        // 在当前实现中，我们直接传递信号而不使用getOutput方法
        double signal = 1.0 * synapse->getWeight(); // 简化处理
        synapse->getPostNeuron()->addInputSignal(signal);
    }
}

const std::vector<std::shared_ptr<Neuron>>& Network::getNeurons() const {
    return neurons_;
}

const std::vector<std::shared_ptr<Synapse>>& Network::getSynapses() const {
    return synapses_;
}

} // namespace neural_network