cmake_minimum_required(VERSION 3.15)
project(ex01_config_validation CXX)

find_package(fmt REQUIRED)
find_package(yaml-cpp REQUIRED)

add_library(config_lib src/config_loader.cpp)
target_include_directories(config_lib PUBLIC include)
target_link_libraries(config_lib PUBLIC fmt::fmt yaml-cpp::yaml-cpp)
target_compile_features(config_lib PUBLIC cxx_std_17)

add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE config_lib)
