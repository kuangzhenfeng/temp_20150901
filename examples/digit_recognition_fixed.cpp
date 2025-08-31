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

// 训练网络识别数字模式
void trainNetwork(const std::vector<std::shared_ptr<Neuron>>& inputLayer,
                  const std::vector<std::shared_ptr<Neuron>>& outputLayer,
                  std::vector<std::shared_ptr<Synapse>>& synapses,
                  const std::vector<DigitPattern>& trainingPatterns,
                  int epochs = 5000) {
    std::cout << "使用监督学习训练网络..." << std::endl;
    
    double learningRate = 0.5;
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double totalError = 0.0;
        
        for (const auto& pattern : trainingPatterns) {
            // 清除之前的所有输入
            for (auto& neuron : inputLayer) {
                neuron->clearInputSignals();
            }
            
            for (auto& neuron : outputLayer) {
                neuron->clearInputSignals();
            }
            
            // 将模式转换为输入
            std::vector<double> inputs = patternToInput(pattern);
            
            // 将输入应用到输入层神经元
            for (size_t i = 0; i < inputs.size() && i < inputLayer.size(); ++i) {
                inputLayer[i]->addInputSignal(inputs[i]);
            }
            
            // 计算输入层输出
            std::vector<double> inputOutputs(inputLayer.size());
            for (size_t i = 0; i < inputLayer.size(); ++i) {
                inputOutputs[i] = inputLayer[i]->computeOutput();
            }
            
            // 通过突触传递信号到输出层
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
            
            // 计算输出层输出
            std::vector<double> outputs(outputLayer.size());
            for (size_t i = 0; i < outputLayer.size(); ++i) {
                outputs[i] = outputLayer[i]->computeOutput();
            }
            
            // 根据目标输出计算误差
            std::vector<double> targetOutputs(outputLayer.size(), 0.1); // 默认低激活
            if (pattern.label < static_cast<int>(targetOutputs.size())) {
                targetOutputs[pattern.label] = 0.9; // 目标数字对应神经元高激活
            }
            
            // 计算总误差
            for (size_t i = 0; i < outputLayer.size(); ++i) {
                double error = targetOutputs[i] - outputs[i];
                totalError += error * error;
            }
            
            // 更新权重（反向传播算法）
            for (size_t i = 0; i < outputLayer.size(); ++i) {
                double error = targetOutputs[i] - outputs[i];
                // 计算delta值（误差项）
                double delta = error * outputs[i] * (1.0 - outputs[i]); // Sigmoid导数
                
                for (auto& synapse : synapses) {
                    if (synapse->getPostNeuron() == outputLayer[i]) {
                        auto pre = synapse->getPreNeuron();
                        auto it = std::find(inputLayer.begin(), inputLayer.end(), pre);
                        if (it != inputLayer.end()) {
                            size_t index = std::distance(inputLayer.begin(), it);
                            // 根据梯度下降法则更新权重
                            double deltaWeight = learningRate * delta * inputOutputs[index];
                            synapse->setWeight(synapse->getWeight() + deltaWeight);
                        }
                    }
                }
            }
        }
        
        // 每1000轮输出一次误差
        if (epoch % 1000 == 0) {
            std::cout << "Epoch " << epoch << ", Error: " << totalError << std::endl;
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
    
    for (auto& neuron : outputLayer) {
        neuron->clearInputSignals();
    }
    
    // 将模式转换为输入
    std::vector<double> input = patternToInput(pattern);
    
    // 将输入应用到输入层神经元
    for (size_t i = 0; i < input.size() && i < inputLayer.size(); ++i) {
        inputLayer[i]->addInputSignal(input[i]);
    }
    
    // 计算输入层输出
    std::vector<double> inputOutputs(inputLayer.size());
    for (size_t i = 0; i < inputLayer.size(); ++i) {
        inputOutputs[i] = inputLayer[i]->computeOutput();
    }
    
    // 通过突触传递信号到输出层
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
    
    // 计算网络输出
    std::vector<double> outputs(outputLayer.size());
    for (size_t i = 0; i < outputLayer.size(); ++i) {
        outputs[i] = outputLayer[i]->computeOutput();
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
    
    std::cout << "  网络输出: ";
    for (size_t i = 0; i < outputs.size(); ++i) {
        std::cout << "数字" << i << "=" << outputs[i] << " ";
    }
    std::cout << std::endl;
    
    return recognizedDigit;
}

int main() {
    std::cout << "=== 手写数字模式识别示例 ===" << std::endl;
    std::cout << "本示例演示如何使用神经网络识别简单的3x3像素手写数字" << std::endl;
    
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
    
    // 创建突触连接（全连接）
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
    
    // 训练网络
    trainNetwork(inputLayer, outputLayer, synapses, trainingPatterns, 5000);
    
    // 测试网络
    std::cout << "\n=== 测试网络 ===" << std::endl;
    
    // 测试数字0
    std::cout << "测试数字0模式:" << std::endl;
    int recognized0 = testPattern(inputLayer, outputLayer, synapses, trainingPatterns[0]);
    std::cout << "识别结果: " << recognized0 << " (正确答案: " << trainingPatterns[0].label << ")" << std::endl;
    
    // 测试数字1
    std::cout << "测试数字1模式:" << std::endl;
    int recognized1 = testPattern(inputLayer, outputLayer, synapses, trainingPatterns[1]);
    std::cout << "识别结果: " << recognized1 << " (正确答案: " << trainingPatterns[1].label << ")" << std::endl;
    
    // 测试一个轻微变化的数字模式
    std::vector<std::vector<int>> variantPattern1 = {
        {0, 1, 0},
        {1, 1, 0},  // 略有变化
        {0, 1, 0}
    };
    DigitPattern variant1(variantPattern1, 1);
    std::cout << "测试变化的数字1模式:" << std::endl;
    int recognizedVariant1 = testPattern(inputLayer, outputLayer, synapses, variant1);
    std::cout << "识别结果: " << recognizedVariant1 << " (正确答案: " << variant1.label << ")" << std::endl;
    
    std::cout << "\n=== 手写数字模式识别示例完成 ===" << std::endl;
    
    return 0;
}