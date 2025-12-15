cmake_minimum_required(VERSION 3.15)
project(ex02_onnx_runtime_answer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(onnxruntime REQUIRED)

add_executable(ort_inference src/main.cpp)
target_link_libraries(ort_inference PRIVATE onnxruntime::onnxruntime)

# Copy data file to build dir
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../data/simple_model.onnx DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
