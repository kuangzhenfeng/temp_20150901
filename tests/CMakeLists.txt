# 测试程序的CMakeLists.txt

# 添加测试程序
add_executable(test_neuron test_neuron.cpp)
add_executable(test_network test_network.cpp)

# 链接主项目库
target_link_libraries(test_neuron ${PROJECT_NAME})
target_link_libraries(test_network ${PROJECT_NAME})

# 设置包含目录
target_include_directories(test_neuron PRIVATE 
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/neuron
    ${CMAKE_SOURCE_DIR}/src/network
)

target_include_directories(test_network PRIVATE 
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/neuron
    ${CMAKE_SOURCE_DIR}/src/network
)

# 设置C++17标准
set_target_properties(test_neuron PROPERTIES 
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

set_target_properties(test_network PROPERTIES 
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)