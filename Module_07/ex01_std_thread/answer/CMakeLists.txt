cmake_minimum_required(VERSION 3.15)
project(ex01_thread_answer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(include)

add_executable(thread_demo src/main.cpp src/parallel_process.cpp)
target_link_libraries(thread_demo PRIVATE ${OpenCV_LIBS})
