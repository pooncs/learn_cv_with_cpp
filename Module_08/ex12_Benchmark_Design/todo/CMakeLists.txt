cmake_minimum_required(VERSION 3.15)
project(ex12_bench_design)

set(CMAKE_CXX_STANDARD 17)

find_package(benchmark REQUIRED)

add_executable(bench_design src/main.cpp)
target_link_libraries(bench_design PRIVATE benchmark::benchmark benchmark::benchmark_main)
