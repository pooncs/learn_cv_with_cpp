cmake_minimum_required(VERSION 3.15)
project(ex02_onnx_runtime CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(onnxruntime REQUIRED)
find_package(GTest REQUIRED)

add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE onnxruntime::onnxruntime)

# Copy data file to build dir
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data/simple_model.onnx DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

enable_testing()
add_executable(test_ort tests/test_ort.cpp)
target_link_libraries(test_ort PRIVATE GTest::gtest_main onnxruntime::onnxruntime)
add_test(NAME test_ort COMMAND test_ort)
