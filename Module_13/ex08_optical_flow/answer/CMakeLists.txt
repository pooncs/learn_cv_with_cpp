cmake_minimum_required(VERSION 3.10)
project(OpticalFlow)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(include)

add_executable(optical_flow src/main.cpp src/optical_flow.cpp)
target_link_libraries(optical_flow ${OpenCV_LIBS})
