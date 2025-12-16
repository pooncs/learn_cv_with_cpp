cmake_minimum_required(VERSION 3.15)
project(ex14_spdlog)

set(CMAKE_CXX_STANDARD 17)

find_package(spdlog REQUIRED)

add_executable(logging_demo src/main.cpp)
target_link_libraries(logging_demo PRIVATE spdlog::spdlog)
