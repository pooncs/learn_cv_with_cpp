cmake_minimum_required(VERSION 3.15)
project(ex10_yolov8_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(GTest REQUIRED)

find_path(TRT_INCLUDE_DIR NvInfer.h HINTS $ENV{TensorRT_ROOT} PATH_SUFFIXES include)
find_library(TRT_LIB nvinfer HINTS $ENV{TensorRT_ROOT} PATH_SUFFIXES lib lib64)

if (TRT_LIB)
    include_directories(${TRT_INCLUDE_DIR})
endif()

add_executable(main src/main.cpp)
if (TRT_LIB)
    target_link_libraries(main PRIVATE OpenCV::opencv ${TRT_LIB} CUDA::cudart)
else()
    target_link_libraries(main PRIVATE OpenCV::opencv)
endif()

enable_testing()
add_executable(test_yolo tests/test_yolo.cpp)
target_link_libraries(test_yolo PRIVATE GTest::gtest_main OpenCV::opencv)
add_test(NAME test_yolo COMMAND test_yolo)
