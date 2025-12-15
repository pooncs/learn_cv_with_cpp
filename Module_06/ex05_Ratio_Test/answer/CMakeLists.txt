cmake_minimum_required(VERSION 3.15)
project(ex05_ratio_answer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(include)

add_executable(ratio_demo src/main.cpp src/ratio_test.cpp)
target_link_libraries(ratio_demo PRIVATE ${OpenCV_LIBS})
