#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <memory>
#include <functional>

namespace neural_network {
    
/**
 * @brief 激活函数类型枚举
 */
enum class ActivationType {
    SIGMOID,
    TANH,
    RELU
};

/**
 * @brief 神经元类（深度学习版本）
 * 
 * 这个类表示深度学习中的神经元节点，包含权重、偏置、激活函数等属性
 */
class Neuron {
public:
    /**
     * @brief 构造函数
     * @param numInputs 输入连接数
     */
    explicit Neuron(size_t numInputs);
    
    /**
     * @brief 析构函数
     */
    virtual ~Neuron() = default;
    
    /**
     * @brief 前向传播计算输出
     * @param inputs 输入值向量
     * @return 输出值
     */
    double forward(const std::vector<double>& inputs);
    
    /**
     * @brief 设置激活函数类型
     * @param type 激活函数类型
     */
    void setActivationFunction(ActivationType type);
    
    /**
     * @brief 设置自定义激活函数
     * @param activation_func 激活函数
     */
    void setActivationFunction(std::function<double(double)> activation_func);
    
    /**
     * @brief 获取权重
     * @return 权重向量
     */
    const std::vector<double>& getWeights() const;
    
    /**
     * @brief 设置权重
     * @param weights 新的权重向量
     */
    void setWeights(const std::vector<double>& weights);
    
    /**
     * @brief 获取偏置
     * @return 偏置值
     */
    double getBias() const;
    
    /**
     * @brief 设置偏置
     * @param bias 新的偏置值
     */
    void setBias(double bias);
    
    /**
     * @brief 获取权重梯度
     * @return 权重梯度向量
     */
    const std::vector<double>& getWeightGradients() const;
    
    /**
     * @brief 获取偏置梯度
     * @return 偏置梯度值
     */
    double getBiasGradient() const;
    
    /**
     * @brief 设置梯度值
     * @param weightGradients 权重梯度向量
     * @param biasGradient 偏置梯度值
     */
    void setGradients(const std::vector<double>& weightGradients, double biasGradient);
    
    /**
     * @brief 更新权重和偏置
     * @param learningRate 学习率
     */
    void updateWeights(double learningRate);
    
    /**
     * @brief 获取最近一次的输出值
     * @return 输出值
     */
    double getOutput() const;
    
    /**
     * @brief 计算激活函数的导数
     * @param output 输出值
     * @return 导数值
     */
    double computeActivationDerivative(double output) const;

protected:
    std::vector<double> weights_;              ///< 连接权重
    double bias_;                              ///< 偏置项
    std::function<double(double)> activation_function_; ///< 激活函数
    ActivationType activation_type_;           ///< 激活函数类型
    
    // 用于反向传播的梯度信息
    std::vector<double> weight_gradients_;     ///< 权重梯度
    double bias_gradient_;                     ///< 偏置梯度
    double output_;                            ///< 最近一次的输出值
    
    /**
     * @brief 初始化指定类型的激活函数
     * @param type 激活函数类型
     */
    void initializeActivationFunction(ActivationType type);
};

} // namespace neural_network

#endif // NEURON_H