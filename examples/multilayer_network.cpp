#include <iostream>
#include <memory>
#include <vector>
#include <cmath>
#include <random>
#include "../src/neuron/neuron.h"
#include "../src/synapse/synapse.h"
#include "../src/network/layer.h"
#include "../src/network/network.h"

using namespace neural_network;

// XOR数据集
struct XorData {
    double input1;
    double input2;
    double output;
    
    XorData(double i1, double i2, double o) : input1(i1), input2(i2), output(o) {}
};

// 训练网络实现XOR功能
void trainXorNetwork(const std::vector<std::shared_ptr<Neuron>>& inputLayer,
                     const std::vector<std::shared_ptr<Neuron>>& hiddenLayer,
                     const std::vector<std::shared_ptr<Neuron>>& outputLayer,
                     const std::vector<std::shared_ptr<Synapse>>& synapses,
                     const std::vector<XorData>& trainingData,
                     int epochs = 10000) {
    std::cout << "训练XOR网络..." << std::endl;
    
    double learningRate = 5.0;
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double totalError = 0.0;
        
        for (const auto& data : trainingData) {
            // 清除之前的所有输入
            for (auto& neuron : inputLayer) {
                neuron->clearInputSignals();
            }
            
            for (auto& neuron : hiddenLayer) {
                neuron->clearInputSignals();
            }
            
            for (auto& neuron : outputLayer) {
                neuron->clearInputSignals();
            }
            
            // 设置输入
            inputLayer[0]->addInputSignal(data.input1);
            inputLayer[1]->addInputSignal(data.input2);
            
            // 计算输入层输出
            std::vector<double> inputOutputs(inputLayer.size());
            for (size_t i = 0; i < inputLayer.size(); ++i) {
                inputOutputs[i] = inputLayer[i]->computeOutput();
            }
            
            // 通过突触传递信号到隐藏层
            for (auto& synapse : synapses) {
                auto pre = synapse->getPreNeuron();
                auto post = synapse->getPostNeuron();
                
                // 检查突触前神经元是否在输入层中
                auto it = std::find(inputLayer.begin(), inputLayer.end(), pre);
                if (it != inputLayer.end()) {
                    size_t index = std::distance(inputLayer.begin(), it);
                    double signal = inputOutputs[index] * synapse->getWeight();
                    post->addInputSignal(signal);
                }
            }
            
            // 计算隐藏层输出
            std::vector<double> hiddenOutputs(hiddenLayer.size());
            for (size_t i = 0; i < hiddenLayer.size(); ++i) {
                hiddenOutputs[i] = hiddenLayer[i]->computeOutput();
            }
            
            // 通过突触传递信号到输出层
            for (auto& synapse : synapses) {
                auto pre = synapse->getPreNeuron();
                auto post = synapse->getPostNeuron();
                
                // 检查突触前神经元是否在隐藏层中
                auto it = std::find(hiddenLayer.begin(), hiddenLayer.end(), pre);
                if (it != hiddenLayer.end()) {
                    size_t index = std::distance(hiddenLayer.begin(), it);
                    double signal = hiddenOutputs[index] * synapse->getWeight();
                    post->addInputSignal(signal);
                }
            }
            
            // 计算输出层输出
            double output = outputLayer[0]->computeOutput();
            
            // 计算误差
            double error = data.output - output;
            totalError += error * error;
            
            // 反向传播更新权重
            // 输出层到隐藏层的权重更新
            double outputDelta = error * output * (1.0 - output); // Sigmoid导数
            
            for (auto& synapse : synapses) {
                if (synapse->getPostNeuron() == outputLayer[0]) {
                    auto pre = synapse->getPreNeuron();
                    auto it = std::find(hiddenLayer.begin(), hiddenLayer.end(), pre);
                    if (it != hiddenLayer.end()) {
                        size_t index = std::distance(hiddenLayer.begin(), it);
                        double deltaWeight = learningRate * outputDelta * hiddenOutputs[index];
                        synapse->setWeight(synapse->getWeight() + deltaWeight);
                    }
                }
            }
            
            // 隐藏层到输入层的权重更新
            std::vector<double> hiddenDeltas(hiddenLayer.size());
            for (size_t i = 0; i < hiddenLayer.size(); ++i) {
                double hiddenOutput = hiddenOutputs[i];
                double downstreamSum = 0.0;
                
                // 计算来自输出层的下游梯度
                for (auto& synapse : synapses) {
                    if (synapse->getPreNeuron() == hiddenLayer[i] && 
                        synapse->getPostNeuron() == outputLayer[0]) {
                        downstreamSum += synapse->getWeight() * outputDelta;
                    }
                }
                
                hiddenDeltas[i] = downstreamSum * hiddenOutput * (1.0 - hiddenOutput);
            }
            
            for (size_t i = 0; i < hiddenLayer.size(); ++i) {
                for (auto& synapse : synapses) {
                    if (synapse->getPostNeuron() == hiddenLayer[i]) {
                        auto pre = synapse->getPreNeuron();
                        auto it = std::find(inputLayer.begin(), inputLayer.end(), pre);
                        if (it != inputLayer.end()) {
                            size_t index = std::distance(inputLayer.begin(), it);
                            double deltaWeight = learningRate * hiddenDeltas[i] * inputOutputs[index];
                            synapse->setWeight(synapse->getWeight() + deltaWeight);
                        }
                    }
                }
            }
        }
        
        // 每2000轮输出一次误差
        if (epoch % 2000 == 0) {
            std::cout << "Epoch " << epoch << ", Error: " << totalError << std::endl;
        }
    }
    
    std::cout << "XOR网络训练完成，训练轮数: " << epochs << std::endl;
}

// 测试XOR网络
double testXorNetwork(const std::vector<std::shared_ptr<Neuron>>& inputLayer,
                      const std::vector<std::shared_ptr<Neuron>>& hiddenLayer,
                      const std::vector<std::shared_ptr<Neuron>>& outputLayer,
                      const std::vector<std::shared_ptr<Synapse>>& synapses,
                      double in1, double in2) {
    // 清除之前的所有输入
    for (auto& neuron : inputLayer) {
        neuron->clearInputSignals();
    }
    
    for (auto& neuron : hiddenLayer) {
        neuron->clearInputSignals();
    }
    
    for (auto& neuron : outputLayer) {
        neuron->clearInputSignals();
    }
    
    // 设置输入
    inputLayer[0]->addInputSignal(in1);
    inputLayer[1]->addInputSignal(in2);
    
    // 计算输入层输出
    std::vector<double> inputOutputs(inputLayer.size());
    for (size_t i = 0; i < inputLayer.size(); ++i) {
        inputOutputs[i] = inputLayer[i]->computeOutput();
    }
    
    // 通过突触传递信号到隐藏层
    for (auto& synapse : synapses) {
        auto pre = synapse->getPreNeuron();
        auto post = synapse->getPostNeuron();
        
        // 检查突触前神经元是否在输入层中
        auto it = std::find(inputLayer.begin(), inputLayer.end(), pre);
        if (it != inputLayer.end()) {
            size_t index = std::distance(inputLayer.begin(), it);
            double signal = inputOutputs[index] * synapse->getWeight();
            post->addInputSignal(signal);
        }
    }
    
    // 计算隐藏层输出
    std::vector<double> hiddenOutputs(hiddenLayer.size());
    for (size_t i = 0; i < hiddenLayer.size(); ++i) {
        hiddenOutputs[i] = hiddenLayer[i]->computeOutput();
    }
    
    // 通过突触传递信号到输出层
    for (auto& synapse : synapses) {
        auto pre = synapse->getPreNeuron();
        auto post = synapse->getPostNeuron();
        
        // 检查突触前神经元是否在隐藏层中
        auto it = std::find(hiddenLayer.begin(), hiddenLayer.end(), pre);
        if (it != hiddenLayer.end()) {
            size_t index = std::distance(hiddenLayer.begin(), it);
            double signal = hiddenOutputs[index] * synapse->getWeight();
            post->addInputSignal(signal);
        }
    }
    
    // 计算输出层输出
    double output = outputLayer[0]->computeOutput();
    
    return output;
}

int main() {
    std::cout << "=== 多层神经网络实现XOR功能 ===" << std::endl;
    std::cout << "本示例演示如何使用多层神经网络解决XOR问题" << std::endl;
    std::cout << "XOR问题是典型的非线性可分问题，单层网络无法解决" << std::endl;
    
    // 创建输入层（2个神经元）
    std::vector<std::shared_ptr<Neuron>> inputLayer;
    for (int i = 0; i < 2; ++i) {
        auto neuron = std::make_shared<Neuron>(i);
        // 使用Sigmoid激活函数
        neuron->setActivationFunction([](double x) -> double {
            return 1.0 / (1.0 + std::exp(-x));
        });
        inputLayer.push_back(neuron);
    }
    
    // 创建隐藏层（4个神经元）
    std::vector<std::shared_ptr<Neuron>> hiddenLayer;
    for (int i = 0; i < 4; ++i) {
        auto neuron = std::make_shared<Neuron>(10 + i);
        neuron->setActivationFunction([](double x) -> double {
            return 1.0 / (1.0 + std::exp(-x));
        });
        hiddenLayer.push_back(neuron);
    }
    
    // 创建输出层（1个神经元）
    std::vector<std::shared_ptr<Neuron>> outputLayer;
    auto outputNeuron = std::make_shared<Neuron>(20);
    outputNeuron->setActivationFunction([](double x) -> double {
        return 1.0 / (1.0 + std::exp(-x));
    });
    outputLayer.push_back(outputNeuron);
    
    // 创建突触连接（全连接）
    std::vector<std::shared_ptr<Synapse>> synapses;
    
    // 输入层到隐藏层的连接
    for (auto& inputNeuron : inputLayer) {
        for (auto& hiddenNeuron : hiddenLayer) {
            // 设置初始权重为小的随机值
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<double> dis(-1.0, 1.0);
            auto synapse = std::make_shared<Synapse>(inputNeuron, hiddenNeuron, dis(gen));
            synapses.push_back(synapse);
        }
    }
    
    // 隐藏层到输出层的连接
    for (auto& hiddenNeuron : hiddenLayer) {
        // 设置初始权重为小的随机值
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(-1.0, 1.0);
        auto synapse = std::make_shared<Synapse>(hiddenNeuron, outputNeuron, dis(gen));
        synapses.push_back(synapse);
    }
    
    std::cout << "创建了包含 " << (inputLayer.size() + hiddenLayer.size() + outputLayer.size()) 
              << " 个神经元的网络" << std::endl;
    std::cout << "创建了 " << synapses.size() << " 个突触连接" << std::endl;
    
    // 创建XOR训练数据
    std::vector<XorData> trainingData;
    trainingData.emplace_back(0.0, 0.0, 0.0);  // 0 XOR 0 = 0
    trainingData.emplace_back(0.0, 1.0, 1.0);  // 0 XOR 1 = 1
    trainingData.emplace_back(1.0, 0.0, 1.0);  // 1 XOR 0 = 1
    trainingData.emplace_back(1.0, 1.0, 0.0);  // 1 XOR 1 = 0
    
    std::cout << "\nXOR训练数据:" << std::endl;
    for (const auto& data : trainingData) {
        std::cout << "  " << data.input1 << " XOR " << data.input2 << " = " << data.output << std::endl;
    }
    
    // 训练网络
    trainXorNetwork(inputLayer, hiddenLayer, outputLayer, synapses, trainingData, 10000);
    
    // 测试网络
    std::cout << "\n=== 测试XOR网络 ===" << std::endl;
    
    for (const auto& data : trainingData) {
        double output = testXorNetwork(inputLayer, hiddenLayer, outputLayer, synapses, 
                                      data.input1, data.input2);
        std::cout << data.input1 << " XOR " << data.input2 << " = " << output 
                  << " (目标: " << data.output << ")" << std::endl;
    }
    
    std::cout << "\n=== 多层神经网络实现XOR功能完成 ===" << std::endl;
    
    return 0;
}