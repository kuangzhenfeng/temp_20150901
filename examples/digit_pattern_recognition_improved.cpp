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

// 简单的3x3像素模式表示数字
struct DigitPattern {
    std::vector<std::vector<int>> pixels;
    int label; // 数字标签
    
    DigitPattern(const std::vector<std::vector<int>>& p, int l) : pixels(p), label(l) {}
};

// 创建数字0的模式
DigitPattern createDigit0() {
    std::vector<std::vector<int>> pattern = {
        {1, 1, 1},
        {1, 0, 1},
        {1, 1, 1}
    };
    return DigitPattern(pattern, 0);
}

// 创建数字1的模式
DigitPattern createDigit1() {
    std::vector<std::vector<int>> pattern = {
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0}
    };
    return DigitPattern(pattern, 1);
}

// 将2D像素模式转换为1D输入向量
std::vector<double> patternToInput(const DigitPattern& pattern) {
    std::vector<double> input;
    for (const auto& row : pattern.pixels) {
        for (int pixel : row) {
            input.push_back(static_cast<double>(pixel));
        }
    }
    return input;
}

// 简单的Hebb学习规则更新权重
void updateWeightsHebb(std::vector<std::shared_ptr<Synapse>>& synapses,
                       const std::vector<double>& inputs,
                       const std::vector<double>& outputs,
                       double learningRate = 0.1) {
    // Hebb学习规则: Δw_ij = η * x_i * y_j
    // 其中 η 是学习率，x_i 是输入，y_j 是输出
    
    size_t inputIdx = 0;
    size_t outputIdx = 0;
    
    for (auto& synapse : synapses) {
        // 简化处理：假设前半部分连接是输入到输出的直接连接
        if (inputIdx < inputs.size() && outputIdx < outputs.size()) {
            double deltaWeight = learningRate * inputs[inputIdx] * outputs[outputIdx];
            synapse->setWeight(synapse->getWeight() + deltaWeight);
            
            outputIdx++;
            if (outputIdx >= outputs.size()) {
                outputIdx = 0;
                inputIdx++;
            }
        }
    }
}

// 训练网络识别数字模式
void trainNetwork(const std::vector<std::shared_ptr<Neuron>>& inputLayer,
                  const std::vector<std::shared_ptr<Neuron>>& outputLayer,
                  std::vector<std::shared_ptr<Synapse>>& synapses,
                  const std::vector<DigitPattern>& trainingPatterns,
                  int epochs = 1000) {
    std::cout << "使用Hebb学习规则训练网络..." << std::endl;
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (const auto& pattern : trainingPatterns) {
            // 清除之前的所有输入
            for (auto& neuron : inputLayer) {
                neuron->clearInputSignals();
            }
            
            // 将模式转换为输入
            std::vector<double> inputs = patternToInput(pattern);
            
            // 将输入应用到输入层神经元
            for (size_t i = 0; i < inputs.size() && i < inputLayer.size(); ++i) {
                inputLayer[i]->addInputSignal(inputs[i]);
            }
            
            // 计算输出层输出
            std::vector<double> outputs;
            for (auto& neuron : outputLayer) {
                outputs.push_back(neuron->computeOutput());
            }
            
            // 根据目标输出调整权重
            std::vector<double> targetOutputs(outputLayer.size(), 0.1); // 默认低激活
            if (pattern.label < static_cast<int>(targetOutputs.size())) {
                targetOutputs[pattern.label] = 0.9; // 目标数字对应神经元高激活
            }
            
            // 使用简单的权重更新规则
            for (size_t i = 0; i < outputLayer.size(); ++i) {
                double error = targetOutputs[i] - outputs[i];
                // 简化的权重更新
                for (size_t j = 0; j < inputLayer.size(); ++j) {
                    // 查找对应的突触
                    for (auto& synapse : synapses) {
                        if (synapse->getPreNeuron() == inputLayer[j] && 
                            synapse->getPostNeuron() == outputLayer[i]) {
                            double deltaWeight = 0.01 * inputs[j] * error;
                            synapse->setWeight(synapse->getWeight() + deltaWeight);
                        }
                    }
                }
            }
        }
    }
    
    std::cout << "网络训练完成，训练轮数: " << epochs << std::endl;
}

// 测试网络对特定模式的识别
int testPattern(const std::vector<std::shared_ptr<Neuron>>& inputLayer,
                const std::vector<std::shared_ptr<Neuron>>& outputLayer,
                const std::vector<std::shared_ptr<Synapse>>& synapses,
                const DigitPattern& pattern) {
    // 清除之前的所有输入
    for (auto& neuron : inputLayer) {
        neuron->clearInputSignals();
    }
    
    // 将模式转换为输入
    std::vector<double> input = patternToInput(pattern);
    
    // 将输入应用到输入层神经元
    for (size_t i = 0; i < input.size() && i < inputLayer.size(); ++i) {
        inputLayer[i]->addInputSignal(input[i]);
    }
    
    // 将输入应用到输入层神经元
    for (size_t i = 0; i < input.size() && i < inputLayer.size(); ++i) {
        inputLayer[i]->addInputSignal(input[i]);
    }
    
    // 添加偏置信号
    for (auto& synapse : synapses) {
        if (synapse->getPreNeuron()->getId() == 999) { // 虚拟偏置神经元
            synapse->getPostNeuron()->addInputSignal(1.0 * synapse->getWeight());
        }
    }
    
    // 计算网络输出
    std::vector<double> outputs;
    for (auto& neuron : outputLayer) {
        outputs.push_back(neuron->computeOutput());
    }
    
    // 识别逻辑：输出值最大的神经元对应的数字就是识别结果
    int recognizedDigit = 0;
    double maxOutput = outputs[0];
    for (size_t i = 1; i < outputs.size(); ++i) {
        if (outputs[i] > maxOutput) {
            maxOutput = outputs[i];
            recognizedDigit = static_cast<int>(i);
        }
    }
    
    return recognizedDigit;
}

// 打印网络权重信息
void printNetworkWeights(const std::vector<std::shared_ptr<Synapse>>& synapses) {
    std::cout << "网络连接权重信息:" << std::endl;
    for (size_t i = 0; i < synapses.size() && i < 10; ++i) { // 只打印前10个
        auto synapse = synapses[i];
        std::cout << "  神经元" << synapse->getPreNeuron()->getId() 
                  << " -> 神经元" << synapse->getPostNeuron()->getId()
                  << " 权重: " << synapse->getWeight() << std::endl;
    }
    if (synapses.size() > 10) {
        std::cout << "  ... 还有 " << synapses.size() - 10 << " 个连接" << std::endl;
    }
}

int main() {
    std::cout << "=== 改进的手写数字模式识别示例 ===" << std::endl;
    
    // 创建输入层（9个神经元对应3x3像素）
    std::vector<std::shared_ptr<Neuron>> inputLayer;
    for (int i = 0; i < 9; ++i) {
        auto neuron = std::make_shared<Neuron>(i);
        // 使用Sigmoid激活函数
        neuron->setActivationFunction([](double x) -> double {
            return 1.0 / (1.0 + std::exp(-x));
        });
        inputLayer.push_back(neuron);
    }
    
    // 创建输出层（2个神经元，分别对应数字0和1）
    std::vector<std::shared_ptr<Neuron>> outputLayer;
    for (int i = 0; i < 2; ++i) {
        auto neuron = std::make_shared<Neuron>(20 + i);
        neuron->setActivationFunction([](double x) -> double {
            return 1.0 / (1.0 + std::exp(-x));
        });
        outputLayer.push_back(neuron);
    }
    
    // 创建突触连接（简化：只连接输入层到输出层）
    std::vector<std::shared_ptr<Synapse>> synapses;
    
    // 输入层到输出层的连接
    for (auto& inputNeuron : inputLayer) {
        for (auto& outputNeuron : outputLayer) {
            // 设置初始权重为小的随机值
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<double> dis(-0.5, 0.5);
            auto synapse = std::make_shared<Synapse>(inputNeuron, outputNeuron, dis(gen));
            synapses.push_back(synapse);
        }
    }
    
    // 添加偏置连接（使用一个虚拟的偏置神经元）
    auto biasNeuron = std::make_shared<Neuron>(999); // 虚拟偏置神经元
    for (auto& outputNeuron : outputLayer) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(-0.5, 0.5);
        auto synapse = std::make_shared<Synapse>(biasNeuron, outputNeuron, dis(gen));
        synapses.push_back(synapse);
    }
    
    std::cout << "创建了包含 " << (inputLayer.size() + outputLayer.size()) << " 个神经元的网络" << std::endl;
    std::cout << "创建了 " << synapses.size() << " 个突触连接" << std::endl;
    
    // 创建训练数据
    std::vector<DigitPattern> trainingPatterns;
    trainingPatterns.push_back(createDigit0());
    trainingPatterns.push_back(createDigit1());
    
    std::cout << "\n训练数据:" << std::endl;
    std::cout << "数字0的模式:" << std::endl;
    for (const auto& row : trainingPatterns[0].pixels) {
        for (int pixel : row) {
            std::cout << (pixel ? "█" : " ");
        }
        std::cout << std::endl;
    }
    
    std::cout << "数字1的模式:" << std::endl;
    for (const auto& row : trainingPatterns[1].pixels) {
        for (int pixel : row) {
            std::cout << (pixel ? "█" : " ");
        }
        std::cout << std::endl;
    }
    
    // 显示初始权重
    std::cout << "\n初始权重:" << std::endl;
    printNetworkWeights(synapses);
    
    // 训练网络
    trainNetwork(inputLayer, outputLayer, synapses, trainingPatterns, 1000);
    
    // 显示训练后权重
    std::cout << "\n训练后权重:" << std::endl;
    printNetworkWeights(synapses);
    
    // 测试网络
    std::cout << "\n=== 测试网络 ===" << std::endl;
    
    // 测试数字0
    int recognized0 = testPattern(inputLayer, outputLayer, synapses, trainingPatterns[0]);
    std::cout << "测试数字0模式，识别结果: " << recognized0 << " (正确答案: " << trainingPatterns[0].label << ")" << std::endl;
    
    // 测试数字1
    int recognized1 = testPattern(inputLayer, outputLayer, synapses, trainingPatterns[1]);
    std::cout << "测试数字1模式，识别结果: " << recognized1 << " (正确答案: " << trainingPatterns[1].label << ")" << std::endl;
    
    // 测试一个轻微变化的数字模式
    std::vector<std::vector<int>> variantPattern1 = {
        {0, 1, 0},
        {1, 1, 0},  // 略有变化
        {0, 1, 0}
    };
    DigitPattern variant1(variantPattern1, 1);
    int recognizedVariant1 = testPattern(inputLayer, outputLayer, synapses, variant1);
    std::cout << "测试变化的数字1模式，识别结果: " << recognizedVariant1 << " (正确答案: " << variant1.label << ")" << std::endl;
    
    std::cout << "\n=== 改进的手写数字模式识别示例完成 ===" << std::endl;
    
    return 0;
}