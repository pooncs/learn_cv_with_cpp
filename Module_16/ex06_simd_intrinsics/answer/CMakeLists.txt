cmake_minimum_required(VERSION 3.15)
project(ex06_simd CXX)

find_package(benchmark REQUIRED)
find_package(fmt REQUIRED)

add_executable(simd_bench src/main.cpp)
target_include_directories(simd_bench PRIVATE include)
target_link_libraries(simd_bench PRIVATE benchmark::benchmark fmt::fmt)
target_compile_features(simd_bench PRIVATE cxx_std_17)

if(MSVC)
    target_compile_options(simd_bench PRIVATE /arch:AVX2)
else()
    target_compile_options(simd_bench PRIVATE -mavx2 -mfma)
endif()
