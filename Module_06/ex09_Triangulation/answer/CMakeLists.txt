cmake_minimum_required(VERSION 3.15)
project(ex09_triangulation_answer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(include)

add_executable(triangulation_demo src/main.cpp src/triangulation.cpp)
target_link_libraries(triangulation_demo PRIVATE ${OpenCV_LIBS})
