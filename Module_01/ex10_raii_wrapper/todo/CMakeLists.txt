cmake_minimum_required(VERSION 3.10)
project(ex10_raii_wrapper)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

# Gather sources
file(GLOB SOURCES "src/*.cpp")

add_executable(main ${SOURCES})

enable_testing()
add_test(NAME ex10_raii_wrapper_Test COMMAND main)
