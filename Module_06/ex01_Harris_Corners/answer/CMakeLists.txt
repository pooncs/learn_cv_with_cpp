cmake_minimum_required(VERSION 3.15)
project(ex01_harris_answer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(include)

add_executable(harris_demo src/main.cpp src/harris.cpp)
target_link_libraries(harris_demo PRIVATE ${OpenCV_LIBS})
