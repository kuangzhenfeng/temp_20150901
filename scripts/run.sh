#!/bin/bash

# 深度学习神经网络框架运行脚本

# 检查参数
if [ $# -eq 0 ]; then
    echo "用法: ./scripts/run.sh [target]"
    echo "可用目标:"
    echo "  main         - 运行主程序"
    echo "  xor          - 运行XOR示例"
    echo "  digit        - 运行手写数字识别示例"
    echo "  test-neuron  - 运行神经元测试"
    echo "  test-network - 运行网络测试"
    echo "  all-tests    - 运行所有测试"
    echo "  clean        - 清理构建目录"
    exit 1
fi

# 检查是否已经构建
if [ ! -d "build" ] || [ ! -f "build/bin/NeuralNetwork_exec" ]; then
    echo "项目尚未构建，请先运行 ./scripts/build.sh"
    exit 1
fi

# 运行指定的目标
case $1 in
    "main")
        echo "运行主程序..."
        ./build/bin/NeuralNetwork_exec
        ;;
    "xor")
        echo "运行XOR示例..."
        ./build/bin/xor_example
        ;;
    "digit")
        echo "运行手写数字识别示例..."
        ./build/bin/digit_recognition
        ;;
    "test-neuron")
        echo "运行神经元测试..."
        ./build/bin/test_neuron_exec
        ;;
    "test-network")
        echo "运行网络测试..."
        ./build/bin/test_network_exec
        ;;
    "all-tests")
        echo "运行所有测试..."
        echo "=== 神经元测试 ==="
        ./build/bin/test_neuron_exec
        echo ""
        echo "=== 网络测试 ==="
        ./build/bin/test_network_exec
        ;;
    "clean")
        echo "清理构建目录..."
        rm -rf build
        echo "清理完成"
        ;;
    *)
        echo "未知目标: $1"
        echo "请使用 ./scripts/run.sh 查看可用目标"
        exit 1
        ;;
esac