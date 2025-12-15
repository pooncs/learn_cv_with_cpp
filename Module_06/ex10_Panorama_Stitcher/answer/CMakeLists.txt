cmake_minimum_required(VERSION 3.15)
project(ex10_stitcher_answer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(include)

add_executable(stitcher_demo src/main.cpp src/stitcher.cpp)
target_link_libraries(stitcher_demo PRIVATE ${OpenCV_LIBS})
