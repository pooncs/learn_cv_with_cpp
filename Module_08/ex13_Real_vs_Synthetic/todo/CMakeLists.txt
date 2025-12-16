cmake_minimum_required(VERSION 3.15)
project(ex13_bench_data)

set(CMAKE_CXX_STANDARD 17)

find_package(benchmark REQUIRED)

add_executable(bench_data src/main.cpp)
target_link_libraries(bench_data PRIVATE benchmark::benchmark benchmark::benchmark_main)
