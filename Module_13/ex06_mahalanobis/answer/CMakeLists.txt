cmake_minimum_required(VERSION 3.15)
project(ex06_mahalanobis_answer)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 REQUIRED)

add_executable(mahalanobis src/main.cpp src/mahalanobis.cpp)
target_include_directories(mahalanobis PRIVATE include)
target_link_libraries(mahalanobis PRIVATE Eigen3::Eigen)
