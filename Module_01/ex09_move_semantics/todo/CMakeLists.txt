cmake_minimum_required(VERSION 3.10)
project(ex09_move_semantics)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

# Gather sources
file(GLOB SOURCES "src/*.cpp")

add_executable(main ${SOURCES})

enable_testing()
add_test(NAME ex09_move_semantics_Test COMMAND main)
