# 示例程序的CMakeLists.txt

# 添加示例程序
add_executable(xor_example xor_example.cpp)
add_executable(digit_recognition digit_recognition.cpp)

# 链接主项目（如果需要）
target_include_directories(xor_example PRIVATE 
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/neuron
    ${CMAKE_SOURCE_DIR}/src/network
)

target_include_directories(digit_recognition PRIVATE 
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/neuron
    ${CMAKE_SOURCE_DIR}/src/network
)

# 链接主项目库
target_link_libraries(xor_example ${PROJECT_NAME})
target_link_libraries(digit_recognition ${PROJECT_NAME})

# 设置C++17标准
set_target_properties(xor_example PROPERTIES 
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

set_target_properties(digit_recognition PROPERTIES 
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON

)
