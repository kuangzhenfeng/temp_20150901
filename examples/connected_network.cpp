#include <iostream>
#include <memory>
#include "../src/neuron/neuron.h"
#include "../src/synapse/synapse.h"
#include "../src/network/layer.h"
#include "../src/network/network.h"

using namespace neural_network;

int main() {
    std::cout << "=== 带连接的神经网络示例 ===" << std::endl;
    
    // 创建神经网络
    auto network = std::make_shared<Network>();
    
    // 创建神经元
    auto neuron1 = std::make_shared<Neuron>(1);
    auto neuron2 = std::make_shared<Neuron>(2);
    auto neuron3 = std::make_shared<Neuron>(3);
    
    // 将神经元添加到网络
    network->addNeuron(neuron1);
    network->addNeuron(neuron2);
    network->addNeuron(neuron3);
    
    // 创建突触连接
    auto synapse12 = std::make_shared<Synapse>(neuron1, neuron2, 0.8);
    auto synapse13 = std::make_shared<Synapse>(neuron1, neuron3, 0.5);
    auto synapse23 = std::make_shared<Synapse>(neuron2, neuron3, 0.3);
    
    // 将突触添加到网络
    network->addSynapse(synapse12);
    network->addSynapse(synapse13);
    network->addSynapse(synapse23);
    
    // 添加输入信号到输入神经元
    neuron1->addInputSignal(1.0);
    
    // 手动模拟信号传递过程
    // neuron1的输出传递给neuron2和neuron3
    double output1 = neuron1->computeOutput();
    neuron2->addInputSignal(output1 * synapse12->getWeight());
    neuron3->addInputSignal(output1 * synapse13->getWeight());
    
    // neuron2的输出也传递给neuron3
    double output2 = neuron2->computeOutput();
    neuron3->addInputSignal(output2 * synapse23->getWeight());
    
    std::cout << "神经元1输出: " << output1 << std::endl;
    std::cout << "神经元2输出: " << output2 << std::endl;
    std::cout << "神经元3输出: " << neuron3->computeOutput() << std::endl;
    
    std::cout << "=== 带连接的神经网络示例运行完成 ===" << std::endl;
    
    return 0;
}