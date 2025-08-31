# 示例程序的CMakeLists.txt

# 添加示例程序
add_executable(simple_network simple_network.cpp)
add_executable(connected_network connected_network.cpp)
add_executable(digit_recognition_fixed digit_recognition_fixed.cpp)
add_executable(multilayer_network multilayer_network.cpp)
add_executable(logic_gates logic_gates.cpp)

# 链接主项目（如果需要）
target_include_directories(simple_network PRIVATE 
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/neuron
    ${CMAKE_SOURCE_DIR}/src/synapse
    ${CMAKE_SOURCE_DIR}/src/network
)

target_include_directories(connected_network PRIVATE 
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/neuron
    ${CMAKE_SOURCE_DIR}/src/synapse
    ${CMAKE_SOURCE_DIR}/src/network
)

target_include_directories(digit_recognition_fixed PRIVATE 
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/neuron
    ${CMAKE_SOURCE_DIR}/src/synapse
    ${CMAKE_SOURCE_DIR}/src/network
)

target_include_directories(multilayer_network PRIVATE 
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/neuron
    ${CMAKE_SOURCE_DIR}/src/synapse
    ${CMAKE_SOURCE_DIR}/src/network
)

target_include_directories(logic_gates PRIVATE 
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/neuron
    ${CMAKE_SOURCE_DIR}/src/synapse
    ${CMAKE_SOURCE_DIR}/src/network
)

# 链接主项目库
target_link_libraries(simple_network BrainNeuralNetwork)
target_link_libraries(connected_network BrainNeuralNetwork)
target_link_libraries(digit_recognition_fixed BrainNeuralNetwork)
target_link_libraries(multilayer_network BrainNeuralNetwork)
target_link_libraries(logic_gates BrainNeuralNetwork)

# 设置C++17标准
set_target_properties(simple_network PROPERTIES 
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

set_target_properties(connected_network PROPERTIES 
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

set_target_properties(digit_recognition_fixed PROPERTIES 
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

set_target_properties(multilayer_network PROPERTIES 
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

set_target_properties(logic_gates PROPERTIES 
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)