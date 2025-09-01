# 深度学习神经网络框架

这是一个使用C++17编写的深度学习神经网络框架。该项目实现了标准的前馈神经网络，支持多层结构、反向传播训练算法等深度学习核心功能。

## 项目特点

- 使用现代C++17标准编写
- 实现标准的前馈神经网络
- 支持多层网络结构
- 实现完整的反向传播训练算法
- 可扩展的架构设计
- 支持不同类型的激活函数和损失函数

## 项目结构

```
src/              # 源代码目录
  ├── neuron/     # 神经元模块
  │   ├── neuron.h
  │   └── neuron.cpp
  ├── network/    # 网络结构模块
  │   ├── layer.h
  │   ├── layer.cpp
  │   ├── network.h
  │   └── network.cpp
tests/            # 测试代码
examples/         # 示例代码
scripts/          # 构建和运行脚本
```

## 构建说明

项目使用CMake构建系统，要求C++17支持。

### 方法1：使用构建脚本（推荐）

```bash
# 构建项目
./scripts/build.sh
```

### 方法2：手动构建

```bash
# 创建构建目录
mkdir build
cd build

# 配置项目
cmake ..

# 编译项目
make
```

## 运行说明

### 方法1：使用运行脚本（推荐）

```bash
# 查看可用选项
./scripts/run.sh

# 运行主程序
./scripts/run.sh main

# 运行XOR示例
./scripts/run.sh xor

# 运行手写数字识别示例
./scripts/run.sh digit

# 运行所有测试
./scripts/run.sh all-tests

# 清理构建目录
./scripts/run.sh clean
```

### 方法2：手动运行

```bash
# 运行主程序
./bin/NeuralNetwork_exec

# 运行XOR示例
./bin/xor_example

# 运行手写数字识别示例
./bin/digit_recognition

# 运行神经元测试
./bin/test_neuron_exec

# 运行网络测试
./bin/test_network_exec
```

## 核心组件

### 主要类

1. `Neuron` (神经元类) - 表示深度学习中的神经元节点，包含权重、偏置和激活函数
2. `Layer` (网络层类) - 组织神经元为层结构，管理层的前向传播
3. `Network` (网络类) - 管理整个神经网络，实现前向传播和训练功能

### 功能特性

- 神经元支持多种激活函数（Sigmoid、Tanh、ReLU）
- 网络支持多层结构
- 支持前向传播计算
- 实现完整的反向传播训练算法
- 支持多种损失函数（均方误差、交叉熵）
- 支持权重更新机制

## 使用示例

### 基本用法

```cpp
#include "network/network.h"
#include "network/layer.h"
#include "neuron/neuron.h"

// 创建网络
neural_network::Network network;

// 添加隐藏层（3个神经元，每个神经元2个输入）
auto hiddenLayer = std::make_shared<neural_network::Layer>(3, 2);
network.addLayer(hiddenLayer);

// 添加输出层（1个神经元，3个输入）
auto outputLayer = std::make_shared<neural_network::Layer>(1, 3);
network.addLayer(outputLayer);

// 前向传播
std::vector<double> inputs = {0.5, 0.3};
std::vector<double> outputs = network.forward(inputs);

// 训练网络
std::vector<double> targets = {1.0};
network.train(inputs, targets, 0.01);
```

### XOR问题示例

```bash
# 编译后运行XOR示例
./bin/xor_example
```

该示例演示了如何使用神经网络解决XOR问题，包含以下步骤：
1. 创建多层神经网络
2. 准备XOR训练数据
3. 训练网络
4. 验证训练结果

### 手写数字识别示例

```bash
# 编译后运行数字识别示例
./bin/digit_recognition
```

该示例演示了如何使用神经网络进行简单的手写数字识别：
1. 创建深层神经网络
2. 使用简单的3x3像素数字数据集
3. 训练网络识别数字0和1
4. 评估识别准确率

## 扩展功能

框架设计支持以下扩展：

1. 实现更多类型的激活函数（如LeakyReLU、ELU等）
2. 添加更多优化算法（如Adam、RMSprop等）
3. 支持更多损失函数
4. 实现正则化技术（如Dropout、L1/L2正则化等）
5. 添加网络保存和加载功能
6. 支持卷积层和池化层（CNN）
7. 支持循环神经网络（RNN、LSTM等）

## 测试

项目包含单元测试程序，可以验证各个组件的功能：

```bash
# 运行神经元测试
./bin/test_neuron_exec

# 运行网络测试
./bin/test_network_exec
```