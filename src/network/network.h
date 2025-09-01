#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <memory>
#include "../neuron/neuron.h"
#include "layer.h"

namespace neural_network {

/**
 * @brief 损失函数类型枚举
 */
enum class LossFunctionType {
    MEAN_SQUARED_ERROR,
    CROSS_ENTROPY
};

class Network {
public:
    Network();
    ~Network();

    /**
     * @brief 添加网络层
     * @param layer 网络层指针
     */
    void addLayer(std::shared_ptr<Layer> layer);
    
    /**
     * @brief 前向传播
     * @param inputs 输入值向量
     * @return 网络输出值向量
     */
    std::vector<double> forward(const std::vector<double>& inputs);
    
    /**
     * @brief 训练网络（反向传播）
     * @param inputs 输入值向量
     * @param targets 目标值向量
     * @param learningRate 学习率
     */
    void train(const std::vector<double>& inputs, const std::vector<double>& targets, double learningRate);
    
    /**
     * @brief 计算损失函数值（均方误差）
     * @param outputs 网络输出
     * @param targets 目标值
     * @return 损失值
     */
    double computeLoss(const std::vector<double>& outputs, const std::vector<double>& targets) const;
    
    /**
     * @brief 获取网络层数
     * @return 层数
     */
    size_t getLayerCount() const;
    
    /**
     * @brief 获取指定层
     * @param index 层索引
     * @return 网络层指针
     */
    std::shared_ptr<Layer> getLayer(size_t index) const;
    
    /**
     * @brief 设置网络损失函数类型
     * @param type 损失函数类型
     */
    void setLossFunctionType(LossFunctionType type);

private:
    std::vector<std::shared_ptr<Layer>> layers_;
    LossFunctionType loss_function_type_;
    
    /**
     * @brief 反向传播算法实现
     * @param targets 目标值
     * @param learningRate 学习率
     */
    void backpropagate(const std::vector<double>& targets, double learningRate);
    
    /**
     * @brief 计算输出层误差
     * @param outputs 网络输出
     * @param targets 目标值
     * @return 误差向量
     */
    std::vector<double> computeOutputLayerErrors(const std::vector<double>& outputs, 
                                                 const std::vector<double>& targets) const;
};

} // namespace neural_network

#endif // NETWORK_H