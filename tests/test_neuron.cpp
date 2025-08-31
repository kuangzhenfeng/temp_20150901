#include <iostream>
#include <cassert>
#include <memory>
#include "../src/neuron/neuron.h"

using namespace neural_network;

int main() {
    std::cout << "开始神经元单元测试..." << std::endl;
    
    // 测试神经元构造
    auto neuron = std::make_shared<Neuron>(1);
    assert(neuron->getId() == 1);
    std::cout << "测试1 - 神经元构造: 通过" << std::endl;
    
    // 测试添加输入信号
    neuron->addInputSignal(1.0);
    neuron->addInputSignal(2.0);
    // 通过访问私有成员的间接方式验证
    double output = neuron->computeOutput();
    assert(output > 0.5); // sigmoid(3.0) 应该大于0.5
    std::cout << "测试2 - 输入信号处理: 通过" << std::endl;
    
    // 测试膜电位获取和设置
    neuron->setMembranePotential(2.0);
    assert(neuron->getMembranePotential() == 2.0);
    std::cout << "测试3 - 膜电位操作: 通过" << std::endl;
    
    // 测试清除输入信号
    neuron->clearInputSignals();
    // 添加一个测试信号以验证清除功能
    neuron->addInputSignal(1.0);
    double output_with_signal = neuron->computeOutput();
    neuron->clearInputSignals();
    double output_after_clear = neuron->computeOutput();
    assert(output_after_clear < output_with_signal); // 清除后输出应该更小
    std::cout << "测试4 - 清除输入信号: 通过" << std::endl;
    
    std::cout << "所有神经元单元测试通过!" << std::endl;
    return 0;
}