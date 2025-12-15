cmake_minimum_required(VERSION 3.15)
project(ex09_par_algo_answer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

# Find TBB if needed (often needed for parallel algos on Linux/Mac, MSVC might have it built-in)
# find_package(TBB) 
# But let's assume Conan handles it or MSVC PPL is used.

add_executable(par_algo_demo src/main.cpp src/parallel_algos.cpp)
if(TBB_FOUND)
    target_link_libraries(par_algo_demo PRIVATE TBB::tbb)
endif()
