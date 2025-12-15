cmake_minimum_required(VERSION 3.15)
project(ex04_ekf_answer)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 REQUIRED)

add_executable(ekf src/main.cpp src/ekf.cpp)
target_include_directories(ekf PRIVATE include)
target_link_libraries(ekf PRIVATE Eigen3::Eigen)
