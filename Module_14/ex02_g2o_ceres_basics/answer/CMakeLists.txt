cmake_minimum_required(VERSION 3.15)
project(ex02_pose_graph CXX)

find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)

add_executable(pose_graph src/main.cpp)
target_include_directories(pose_graph PRIVATE include)
target_link_libraries(pose_graph PRIVATE Eigen3::Eigen fmt::fmt)
target_compile_features(pose_graph PRIVATE cxx_std_17)
