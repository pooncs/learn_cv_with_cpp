cmake_minimum_required(VERSION 3.15)
project(ex06_flann_answer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(include)

add_executable(flann_demo src/main.cpp src/flann_matcher.cpp)
target_link_libraries(flann_demo PRIVATE ${OpenCV_LIBS})
