#ifndef SYNAPSE_H
#define SYNAPSE_H

#include <memory>

namespace neural_network {
    
// 前向声明
class Neuron;

/**
 * @brief 突触类
 * 
 * 表示两个神经元之间的连接，负责信号从一个神经元传递到另一个神经元
 */
class Synapse {
public:
    /**
     * @brief 构造函数
     * @param pre 突触前神经元
     * @param post 突触后神经元
     * @param weight 连接权重
     */
    Synapse(std::shared_ptr<Neuron> pre, std::shared_ptr<Neuron> post, double weight);
    
    /**
     * @brief 析构函数
     */
    ~Synapse() = default;
    
    /**
     * @brief 获取突触前神经元
     * @return 突触前神经元指针
     */
    std::shared_ptr<Neuron> getPreNeuron() const;
    
    /**
     * @brief 获取突触后神经元
     * @return 突触后神经元指针
     */
    std::shared_ptr<Neuron> getPostNeuron() const;
    
    /**
     * @brief 获取连接权重
     * @return 连接权重值
     */
    double getWeight() const;
    
    /**
     * @brief 设置连接权重
     * @param weight 新的权重值
     */
    void setWeight(double weight);
    
    /**
     * @brief 传递信号
     * 
     * 将突触前神经元的信号乘以权重后传递给突触后神经元
     */
    void transmit();
    
private:
    std::shared_ptr<Neuron> pre_neuron_;   ///< 突触前神经元
    std::shared_ptr<Neuron> post_neuron_;  ///< 突触后神经元
    double weight_;                       ///< 连接权重
};

} // namespace neural_network

#endif // SYNAPSE_H