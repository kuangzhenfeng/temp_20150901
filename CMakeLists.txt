cmake_minimum_required(VERSION 3.10)
project(BrainNeuralNetwork VERSION 1.0.0 LANGUAGES CXX)

# 设置C++标准为C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 编译器选项
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -pedantic)
endif()

# 设置源文件
set(SOURCES
    src/neuron/neuron.cpp
    src/synapse/synapse.cpp
    src/network/layer.cpp
    src/network/network.cpp
)

# 设置头文件目录
include_directories(
    src
    src/neuron
    src/synapse
    src/network
)

# 创建库
add_library(${PROJECT_NAME} ${SOURCES})

# 创建可执行文件
add_executable(${PROJECT_NAME}_exec src/main.cpp)
target_link_libraries(${PROJECT_NAME}_exec ${PROJECT_NAME})

# 设置输出目录
set_target_properties(${PROJECT_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)
set_target_properties(${PROJECT_NAME}_exec PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 添加示例子目录
add_subdirectory(examples)
add_subdirectory(tests)