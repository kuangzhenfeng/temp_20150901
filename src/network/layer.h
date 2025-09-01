#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <memory>
#include "../neuron/neuron.h"

namespace neural_network {
    
/**
 * @brief 网络层类（深度学习版本）
 * 
 * 表示神经网络中的一层，包含多个神经元
 */
class Layer {
public:
    /**
     * @brief 构造函数
     * @param numNeurons 该层神经元数量
     * @param numInputs 每个神经元的输入数量
     */
    Layer(size_t numNeurons, size_t numInputs);
    
    /**
     * @brief 析构函数
     */
    ~Layer() = default;
    
    /**
     * @brief 前向传播
     * @param inputs 输入值向量
     * @return 该层输出值向量
     */
    std::vector<double> forward(const std::vector<double>& inputs);
    
    /**
     * @brief 获取该层所有神经元
     * @return 神经元指针向量
     */
    const std::vector<std::shared_ptr<Neuron>>& getNeurons() const;
    
    /**
     * @brief 获取该层神经元数量
     * @return 神经元数量
     */
    size_t size() const;
    
    /**
     * @brief 设置该层所有神经元的梯度
     * @param weightGradients 权重梯度矩阵
     * @param biasGradients 偏置梯度向量
     */
    void setGradients(const std::vector<std::vector<double>>& weightGradients, 
                      const std::vector<double>& biasGradients);
    
    /**
     * @brief 更新该层所有神经元的权重
     * @param learningRate 学习率
     */
    void updateWeights(double learningRate);
    
    /**
     * @brief 获取最近一次的输入
     * @return 输入值向量
     */
    const std::vector<double>& getLastInputs() const;
    
    /**
     * @brief 获取最近一次的输出
     * @return 输出值向量
     */
    const std::vector<double>& getLastOutputs() const;

private:
    std::vector<std::shared_ptr<Neuron>> neurons_; ///< 层中的神经元
    std::vector<double> last_inputs_;              ///< 最近一次的输入
    std::vector<double> last_outputs_;             ///< 最近一次的输出
};

} // namespace neural_network

#endif // LAYER_H