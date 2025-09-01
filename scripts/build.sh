#!/bin/bash

# 深度学习神经网络框架构建脚本

set -e  # 遇到错误时停止执行

echo "=== 深度学习神经网络框架构建脚本 ==="

# 检查是否在项目根目录
if [ ! -f "CMakeLists.txt" ]; then
    echo "错误: 请在项目根目录运行此脚本"
    exit 1
fi

# 创建构建目录
echo "1. 创建构建目录..."
mkdir -p build
cd build

# 配置项目
echo "2. 配置项目..."
cmake ..

# 编译项目
echo "3. 编译项目..."
make -j$(nproc)

echo "4. 构建完成!"

# 显示生成的可执行文件
echo ""
echo "=== 生成的可执行文件 ==="
echo "主程序: ./bin/NeuralNetwork_exec"
echo "XOR示例: ./bin/xor_example"
echo "数字识别示例: ./bin/digit_recognition"
echo "神经元测试: ./bin/test_neuron_exec"
echo "网络测试: ./bin/test_network_exec"

echo ""
echo "=== 运行示例 ==="
echo "运行主程序: ./bin/NeuralNetwork_exec"
echo "运行XOR示例: ./bin/xor_example"
echo "运行数字识别示例: ./bin/digit_recognition"
echo "运行神经元测试: ./bin/test_neuron_exec"
echo "运行网络测试: ./bin/test_network_exec"