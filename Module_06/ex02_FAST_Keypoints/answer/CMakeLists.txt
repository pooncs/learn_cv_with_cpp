cmake_minimum_required(VERSION 3.15)
project(ex02_fast_answer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(include)

add_executable(fast_demo src/main.cpp src/fast.cpp)
target_link_libraries(fast_demo PRIVATE ${OpenCV_LIBS})
