#include "layer.h"
#include "../neuron/neuron.h"

namespace neural_network {
    
Layer::Layer(int id) : id_(id) {
}

void Layer::addNeuron(std::shared_ptr<Neuron> neuron) {
    neurons_.push_back(neuron);
}

const std::vector<std::shared_ptr<Neuron>>& Layer::getNeurons() const {
    return neurons_;
}

int Layer::getId() const {
    return id_;
}

size_t Layer::size() const {
    return neurons_.size();
}

} // namespace neural_network