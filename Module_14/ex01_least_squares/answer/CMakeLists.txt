cmake_minimum_required(VERSION 3.15)
project(ex01_ls CXX)

find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)

add_executable(ls_demo src/main.cpp)
target_include_directories(ls_demo PRIVATE include)
target_link_libraries(ls_demo PRIVATE Eigen3::Eigen fmt::fmt)
target_compile_features(ls_demo PRIVATE cxx_std_17)
