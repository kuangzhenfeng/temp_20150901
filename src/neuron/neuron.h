#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <memory>
#include <functional>

namespace neural_network {
    
/**
 * @brief 神经元基类
 * 
 * 这个类表示一个基本的神经元，包含输入、输出和激活函数等基本属性
 */
class Neuron {
public:
    /**
     * @brief 构造函数
     * @param id 神经元唯一标识符
     */
    explicit Neuron(int id);
    
    /**
     * @brief 析构函数
     */
    virtual ~Neuron() = default;
    
    /**
     * @brief 获取神经元ID
     * @return 神经元ID
     */
    int getId() const;
    
    /**
     * @brief 添加输入信号
     * @param signal 输入信号值
     */
    void addInputSignal(double signal);
    
    /**
     * @brief 计算输出信号
     * @return 输出信号值
     */
    virtual double computeOutput();
    
    /**
     * @brief 清除输入信号
     */
    void clearInputSignals();
    
    /**
     * @brief 设置激活函数
     * @param activation_func 激活函数
     */
    void setActivationFunction(std::function<double(double)> activation_func);
    
    /**
     * @brief 获取当前膜电位
     * @return 膜电位值
     */
    double getMembranePotential() const;
    
    /**
     * @brief 设置膜电位
     * @param potential 膜电位值
     */
    void setMembranePotential(double potential);
    
protected:
    int id_;                                    ///< 神经元唯一标识符
    std::vector<double> input_signals_;        ///< 输入信号列表
    double membrane_potential_;                ///< 膜电位
    double threshold_;                         ///< 激活阈值
    std::function<double(double)> activation_function_; ///< 激活函数
};

} // namespace neural_network

#endif // NEURON_H