cmake_minimum_required(VERSION 3.15)
project(ex02_pose_graph_todo CXX)

find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)

add_executable(main src/main.cpp)
target_include_directories(main PRIVATE include)
target_link_libraries(main PRIVATE Eigen3::Eigen fmt::fmt)
target_compile_features(main PRIVATE cxx_std_17)
