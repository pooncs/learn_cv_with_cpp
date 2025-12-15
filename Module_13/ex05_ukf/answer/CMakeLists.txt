cmake_minimum_required(VERSION 3.15)
project(ex05_ukf_answer)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 REQUIRED)

add_executable(ukf src/main.cpp src/ukf.cpp)
target_include_directories(ukf PRIVATE include)
target_link_libraries(ukf PRIVATE Eigen3::Eigen)
