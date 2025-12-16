cmake_minimum_required(VERSION 3.15)
project(ex07_preprocessing_answer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(preproc src/main.cpp)
target_link_libraries(preproc PRIVATE OpenCV::opencv)
