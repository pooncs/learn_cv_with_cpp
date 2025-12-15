cmake_minimum_required(VERSION 3.15)
project(ex08_epipolar_answer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(include)

add_executable(epipolar_demo src/main.cpp src/epipolar.cpp)
target_link_libraries(epipolar_demo PRIVATE ${OpenCV_LIBS})
