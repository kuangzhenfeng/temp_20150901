#include <iostream>
#include <memory>
#include <vector>
#include <cmath>
#include "../src/neuron/neuron.h"
#include "../src/synapse/synapse.h"
#include "../src/network/layer.h"
#include "../src/network/network.h"

using namespace neural_network;

// 为神经元设置Sigmoid激活函数
void setSigmoidActivation(std::shared_ptr<Neuron> neuron) {
    neuron->setActivationFunction([](double x) -> double {
        return 1.0 / (1.0 + std::exp(-x));
    });
}

// 训练AND门
void trainANDGate(std::shared_ptr<Neuron> /*input1*/, 
                  std::shared_ptr<Neuron> /*input2*/, 
                  std::shared_ptr<Neuron> output,
                  std::shared_ptr<Synapse> synapse1,
                  std::shared_ptr<Synapse> synapse2) {
    std::cout << "训练AND门..." << std::endl;
    
    // AND门真值表:
    // 0 0 -> 0
    // 0 1 -> 0
    // 1 0 -> 0
    // 1 1 -> 1
    
    // 设置权重和偏置以实现AND功能
    // 权重设置为较大正值，偏置设置为负值
    synapse1->setWeight(10.0);
    synapse2->setWeight(10.0);
    
    // 为输出神经元设置偏置(通过添加一个固定输入)
    output->addInputSignal(-15.0); // 相当于偏置
    
    std::cout << "AND门训练完成" << std::endl;
}

// 训练OR门
void trainORGate(std::shared_ptr<Neuron> /*input1*/, 
                 std::shared_ptr<Neuron> /*input2*/, 
                 std::shared_ptr<Neuron> output,
                 std::shared_ptr<Synapse> synapse1,
                 std::shared_ptr<Synapse> synapse2) {
    std::cout << "训练OR门..." << std::endl;
    
    // OR门真值表:
    // 0 0 -> 0
    // 0 1 -> 1
    // 1 0 -> 1
    // 1 1 -> 1
    
    // 设置权重和偏置以实现OR功能
    synapse1->setWeight(10.0);
    synapse2->setWeight(10.0);
    
    // 为输出神经元设置偏置
    output->addInputSignal(-5.0); // 相当于偏置
    
    std::cout << "OR门训练完成" << std::endl;
}

// 测试逻辑门
void testLogicGate(const std::string& gateName,
                   std::shared_ptr<Neuron> input1, 
                   std::shared_ptr<Neuron> input2, 
                   std::shared_ptr<Neuron> output,
                   std::shared_ptr<Synapse> synapse1,
                   std::shared_ptr<Synapse> synapse2,
                   double in1, double in2) {
    // 清除输入信号
    input1->clearInputSignals();
    input2->clearInputSignals();
    output->clearInputSignals();
    
    // 添加输入信号
    input1->addInputSignal(in1);
    input2->addInputSignal(in2);
    
    // 重新添加偏置（每次测试都需要）
    if (gateName == "AND") {
        output->addInputSignal(-15.0);
    } else if (gateName == "OR") {
        output->addInputSignal(-5.0);
    }
    
    // 手动计算通过突触的信号传递
    double signal1 = in1 * synapse1->getWeight();
    double signal2 = in2 * synapse2->getWeight();
    output->addInputSignal(signal1);
    output->addInputSignal(signal2);
    
    // 计算输出
    double result = output->computeOutput();
    
    std::cout << gateName << "门测试: " << in1 << " " << in2 << " -> " 
              << (result > 0.5 ? 1.0 : 0.0) << " (原始输出: " << result << ")" << std::endl;
}

int main() {
    std::cout << "=== 逻辑门实现示例 ===" << std::endl;
    std::cout << "本示例演示如何使用神经网络实现基本的数字逻辑功能" << std::endl;
    
    // 创建AND门网络
    std::cout << "\n--- AND门 ---" << std::endl;
    auto andInput1 = std::make_shared<Neuron>(1);
    auto andInput2 = std::make_shared<Neuron>(2);
    auto andOutput = std::make_shared<Neuron>(3);
    
    setSigmoidActivation(andInput1);
    setSigmoidActivation(andInput2);
    setSigmoidActivation(andOutput);
    
    auto andSynapse1 = std::make_shared<Synapse>(andInput1, andOutput, 1.0);
    auto andSynapse2 = std::make_shared<Synapse>(andInput2, andOutput, 1.0);
    
    trainANDGate(andInput1, andInput2, andOutput, andSynapse1, andSynapse2);
    
    // 测试AND门
    testLogicGate("AND", andInput1, andInput2, andOutput, andSynapse1, andSynapse2, 0.0, 0.0);
    testLogicGate("AND", andInput1, andInput2, andOutput, andSynapse1, andSynapse2, 0.0, 1.0);
    testLogicGate("AND", andInput1, andInput2, andOutput, andSynapse1, andSynapse2, 1.0, 0.0);
    testLogicGate("AND", andInput1, andInput2, andOutput, andSynapse1, andSynapse2, 1.0, 1.0);
    
    // 创建OR门网络
    std::cout << "\n--- OR门 ---" << std::endl;
    auto orInput1 = std::make_shared<Neuron>(4);
    auto orInput2 = std::make_shared<Neuron>(5);
    auto orOutput = std::make_shared<Neuron>(6);
    
    setSigmoidActivation(orInput1);
    setSigmoidActivation(orInput2);
    setSigmoidActivation(orOutput);
    
    auto orSynapse1 = std::make_shared<Synapse>(orInput1, orOutput, 1.0);
    auto orSynapse2 = std::make_shared<Synapse>(orInput2, orOutput, 1.0);
    
    trainORGate(orInput1, orInput2, orOutput, orSynapse1, orSynapse2);
    
    // 测试OR门
    testLogicGate("OR", orInput1, orInput2, orOutput, orSynapse1, orSynapse2, 0.0, 0.0);
    testLogicGate("OR", orInput1, orInput2, orOutput, orSynapse1, orSynapse2, 0.0, 1.0);
    testLogicGate("OR", orInput1, orInput2, orOutput, orSynapse1, orSynapse2, 1.0, 0.0);
    testLogicGate("OR", orInput1, orInput2, orOutput, orSynapse1, orSynapse2, 1.0, 1.0);
    
    // 创建一个组合逻辑电路: (A AND B) OR (NOT A AND C)
    std::cout << "\n--- 组合逻辑电路: (A AND B) OR (NOT A AND C) ---" << std::endl;
    
    // 输入神经元
    auto circuitA = std::make_shared<Neuron>(10);
    auto circuitB = std::make_shared<Neuron>(11);
    auto circuitC = std::make_shared<Neuron>(12);
    
    setSigmoidActivation(circuitA);
    setSigmoidActivation(circuitB);
    setSigmoidActivation(circuitC);
    
    // NOT A 神经元 (通过负权重实现)
    auto notA = std::make_shared<Neuron>(13);
    setSigmoidActivation(notA);
    auto synapseANotA = std::make_shared<Synapse>(circuitA, notA, -10.0);
    notA->addInputSignal(5.0); // 偏置，使NOT功能更准确
    
    // AND1: A AND B
    auto and1Output = std::make_shared<Neuron>(14);
    setSigmoidActivation(and1Output);
    auto synapseAAnd1 = std::make_shared<Synapse>(circuitA, and1Output, 10.0);
    auto synapseBAnd1 = std::make_shared<Synapse>(circuitB, and1Output, 10.0);
    and1Output->addInputSignal(-15.0); // AND门偏置
    
    // AND2: NOT A AND C
    auto and2Output = std::make_shared<Neuron>(15);
    setSigmoidActivation(and2Output);
    auto synapseNotAAnd2 = std::make_shared<Synapse>(notA, and2Output, 10.0);
    auto synapseCAnd2 = std::make_shared<Synapse>(circuitC, and2Output, 10.0);
    and2Output->addInputSignal(-15.0); // AND门偏置
    
    // OR: AND1 OR AND2
    auto finalOrOutput = std::make_shared<Neuron>(16);
    setSigmoidActivation(finalOrOutput);
    auto synapseAnd1Or = std::make_shared<Synapse>(and1Output, finalOrOutput, 10.0);
    auto synapseAnd2Or = std::make_shared<Synapse>(and2Output, finalOrOutput, 10.0);
    finalOrOutput->addInputSignal(-5.0); // OR门偏置
    
    // 测试组合逻辑电路
    std::cout << "测试组合逻辑电路: (A AND B) OR (NOT A AND C)" << std::endl;
    
    // 测试所有可能的输入组合
    std::vector<std::vector<double>> testCases = {
        {0.0, 0.0, 0.0},  // 0
        {0.0, 0.0, 1.0},  // 1
        {0.0, 1.0, 0.0},  // 0
        {0.0, 1.0, 1.0},  // 1
        {1.0, 0.0, 0.0},  // 0
        {1.0, 0.0, 1.0},  // 0
        {1.0, 1.0, 0.0},  // 1
        {1.0, 1.0, 1.0}   // 1
    };
    
    for (const auto& testCase : testCases) {
        double A = testCase[0];
        double B = testCase[1];
        double C = testCase[2];
        
        // 清除所有输入信号
        circuitA->clearInputSignals();
        circuitB->clearInputSignals();
        circuitC->clearInputSignals();
        notA->clearInputSignals();
        and1Output->clearInputSignals();
        and2Output->clearInputSignals();
        finalOrOutput->clearInputSignals();
        
        // 设置输入
        circuitA->addInputSignal(A);
        circuitB->addInputSignal(B);
        circuitC->addInputSignal(C);
        
        // 计算NOT A
        notA->addInputSignal(A * synapseANotA->getWeight());
        notA->addInputSignal(5.0); // 偏置
        double notAResult = notA->computeOutput();
        
        // 计算AND1: A AND B
        and1Output->addInputSignal(A * synapseAAnd1->getWeight());
        and1Output->addInputSignal(B * synapseBAnd1->getWeight());
        and1Output->addInputSignal(-15.0); // 偏置
        double and1Result = and1Output->computeOutput();
        
        // 计算AND2: NOT A AND C
        and2Output->addInputSignal(notAResult * synapseNotAAnd2->getWeight());
        and2Output->addInputSignal(C * synapseCAnd2->getWeight());
        and2Output->addInputSignal(-15.0); // 偏置
        double and2Result = and2Output->computeOutput();
        
        // 计算最终OR: AND1 OR AND2
        finalOrOutput->addInputSignal(and1Result * synapseAnd1Or->getWeight());
        finalOrOutput->addInputSignal(and2Result * synapseAnd2Or->getWeight());
        finalOrOutput->addInputSignal(-5.0); // 偏置
        double result = finalOrOutput->computeOutput();
        
        std::cout << "输入: A=" << A << " B=" << B << " C=" << C 
                  << " -> 输出: " << (result > 0.5 ? 1.0 : 0.0) 
                  << " (原始值: " << result << ")" << std::endl;
    }
    
    std::cout << "\n=== 逻辑门实现示例完成 ===" << std::endl;
    
    return 0;
}