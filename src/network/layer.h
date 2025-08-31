#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <memory>

namespace neural_network {
    
// 前向声明
class Neuron;

/**
 * @brief 网络层类
 * 
 * 表示神经网络中的一层，包含多个神经元
 */
class Layer {
public:
    /**
     * @brief 构造函数
     * @param id 层的唯一标识符
     */
    explicit Layer(int id);
    
    /**
     * @brief 析构函数
     */
    ~Layer() = default;
    
    /**
     * @brief 添加神经元到层中
     * @param neuron 神经元指针
     */
    void addNeuron(std::shared_ptr<Neuron> neuron);
    
    /**
     * @brief 获取层中所有神经元
     * @return 神经元指针向量
     */
    const std::vector<std::shared_ptr<Neuron>>& getNeurons() const;
    
    /**
     * @brief 获取层ID
     * @return 层ID
     */
    int getId() const;
    
    /**
     * @brief 获取层中神经元数量
     * @return 神经元数量
     */
    size_t size() const;
    
private:
    int id_;                                       ///< 层ID
    std::vector<std::shared_ptr<Neuron>> neurons_; ///< 层中的神经元
};

} // namespace neural_network

#endif // LAYER_H