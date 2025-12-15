cmake_minimum_required(VERSION 3.15)
project(ex02_astar CXX)

find_package(fmt REQUIRED)

add_executable(astar_demo src/main.cpp)
target_include_directories(astar_demo PRIVATE include)
target_link_libraries(astar_demo PRIVATE fmt::fmt)
target_compile_features(astar_demo PRIVATE cxx_std_17)
