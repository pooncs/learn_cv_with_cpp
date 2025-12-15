cmake_minimum_required(VERSION 3.15)
project(ex04_bf_answer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(include)

add_executable(bf_demo src/main.cpp src/matcher.cpp)
target_link_libraries(bf_demo PRIVATE ${OpenCV_LIBS})
