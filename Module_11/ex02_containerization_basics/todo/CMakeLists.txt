cmake_minimum_required(VERSION 3.15)
project(ex02_docker_todo CXX)

find_package(fmt REQUIRED)

add_executable(docker_app src/main.cpp)
target_link_libraries(docker_app PRIVATE fmt::fmt)
target_compile_features(docker_app PRIVATE cxx_std_17)
