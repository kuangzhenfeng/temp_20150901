#include <iostream>
#include <memory>
#include "../src/neuron/neuron.h"
#include "../src/synapse/synapse.h"
#include "../src/network/layer.h"
#include "../src/network/network.h"

using namespace neural_network;

int main() {
    std::cout << "=== 简单神经网络示例 ===" << std::endl;
    
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
    
    // 添加输入信号
    neuron1->addInputSignal(1.0);
    
    // 直接计算输出（绕过网络更新逻辑，因为我们没有设置突触连接）
    std::cout << "神经元1输出: " << neuron1->computeOutput() << std::endl;
    // neuron2和neuron3没有输入信号，所以输出应该是sigmoid(0)=0.5
    std::cout << "神经元2输出: " << neuron2->computeOutput() << std::endl;
    std::cout << "神经元3输出: " << neuron3->computeOutput() << std::endl;
    
    std::cout << "=== 简单神经网络示例运行完成 ===" << std::endl;
    
    return 0;
}