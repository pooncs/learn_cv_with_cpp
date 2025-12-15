cmake_minimum_required(VERSION 3.10)
project(ex03_props LANGUAGES CXX)

add_executable(my_app src/main.cpp)

# Require C++17 features
target_compile_features(my_app PRIVATE cxx_std_17)

# Define a macro
target_compile_definitions(my_app PRIVATE MY_FEATURE)
