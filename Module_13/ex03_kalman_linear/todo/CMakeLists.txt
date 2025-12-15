cmake_minimum_required(VERSION 3.15)
project(ex03_kalman_linear_todo)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 REQUIRED)

add_subdirectory(tests)

add_executable(kalman_linear src/main.cpp src/kalman.cpp)
target_include_directories(kalman_linear PRIVATE include)
target_link_libraries(kalman_linear PRIVATE Eigen3::Eigen)
