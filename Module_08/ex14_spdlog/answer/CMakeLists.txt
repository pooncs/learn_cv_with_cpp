cmake_minimum_required(VERSION 3.15)
project(ex14_spdlog_answer)

set(CMAKE_CXX_STANDARD 17)

find_package(spdlog REQUIRED)

add_executable(logging_demo_answer src/main.cpp)
target_link_libraries(logging_demo_answer PRIVATE spdlog::spdlog)
