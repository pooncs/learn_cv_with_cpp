cmake_minimum_required(VERSION 3.15)
project(ex15_glog_answer)

set(CMAKE_CXX_STANDARD 17)

find_package(glog REQUIRED)

add_executable(glog_demo_answer src/main.cpp)
target_link_libraries(glog_demo_answer PRIVATE glog::glog)
