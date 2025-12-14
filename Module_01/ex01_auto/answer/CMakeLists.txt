cmake_minimum_required(VERSION 3.15)
project(ex01_auto)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(fmt REQUIRED)
find_package(GTest REQUIRED)

include_directories(include)

add_executable(main src/main.cpp)
target_link_libraries(main fmt::fmt)

enable_testing()
add_test(NAME MainTest COMMAND main)
