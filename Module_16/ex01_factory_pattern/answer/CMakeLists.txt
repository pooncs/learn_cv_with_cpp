cmake_minimum_required(VERSION 3.15)
project(ex01_factory CXX)

find_package(fmt REQUIRED)

add_executable(factory_demo src/main.cpp)
target_include_directories(factory_demo PRIVATE include)
target_link_libraries(factory_demo PRIVATE fmt::fmt)
target_compile_features(factory_demo PRIVATE cxx_std_17)
