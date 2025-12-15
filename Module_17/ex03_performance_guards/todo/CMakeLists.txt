cmake_minimum_required(VERSION 3.15)
project(ex03_perf CXX)

find_package(benchmark REQUIRED)
find_package(fmt REQUIRED)

add_executable(bench_process bench/bench_process.cpp)
target_include_directories(bench_process PRIVATE src)
target_link_libraries(bench_process PRIVATE benchmark::benchmark fmt::fmt)
