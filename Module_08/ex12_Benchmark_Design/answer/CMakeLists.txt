cmake_minimum_required(VERSION 3.15)
project(ex12_bench_design_answer)

set(CMAKE_CXX_STANDARD 17)

find_package(benchmark REQUIRED)

add_executable(bench_design_answer src/main.cpp)
target_link_libraries(bench_design_answer PRIVATE benchmark::benchmark benchmark::benchmark_main)
