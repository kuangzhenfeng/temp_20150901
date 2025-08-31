#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <memory>
#include "../neuron/neuron.h"
#include "../synapse/synapse.h"

namespace neural_network {

class Network {
public:
    Network();
    ~Network();

    void addNeuron(std::shared_ptr<Neuron> neuron);
    void removeNeuron(std::shared_ptr<Neuron> neuron);
    
    void addSynapse(std::shared_ptr<Synapse> synapse);
    void removeSynapse(std::shared_ptr<Synapse> synapse);

    void update();

    const std::vector<std::shared_ptr<Neuron>>& getNeurons() const;
    const std::vector<std::shared_ptr<Synapse>>& getSynapses() const;

private:
    std::vector<std::shared_ptr<Neuron>> neurons_;
    std::vector<std::shared_ptr<Synapse>> synapses_;
};

} // namespace neural_network

#endif // NETWORK_H