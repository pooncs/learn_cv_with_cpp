cmake_minimum_required(VERSION 3.15)
project(ex11_benchmark_answer)

set(CMAKE_CXX_STANDARD 17)

find_package(benchmark REQUIRED)

add_executable(bench_demo src/main.cpp)
target_link_libraries(bench_demo PRIVATE benchmark::benchmark benchmark::benchmark_main)
