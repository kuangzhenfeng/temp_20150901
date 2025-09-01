# 测试程序的CMakeLists.txt

# 添加测试程序
add_executable(test_neuron_exec test_neuron.cpp)

# 链接主项目库
target_link_libraries(test_neuron_exec NeuralNetwork)

# 设置包含目录
target_include_directories(test_neuron_exec PRIVATE 
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/neuron
    ${CMAKE_SOURCE_DIR}/src/synapse
    ${CMAKE_SOURCE_DIR}/src/network
)

# 设置C++17标准
set_target_properties(test_neuron_exec PROPERTIES 
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 添加网络测试程序
add_executable(test_network_exec test_network.cpp)
target_link_libraries(test_network_exec NeuralNetwork)
target_include_directories(test_network_exec PRIVATE 
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/neuron
    ${CMAKE_SOURCE_DIR}/src/synapse
    ${CMAKE_SOURCE_DIR}/src/network
)
set_target_properties(test_network_exec PROPERTIES 
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)