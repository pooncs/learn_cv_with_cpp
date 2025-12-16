cmake_minimum_required(VERSION 3.15)
project(ex03_include_dirs_answer)

set(CMAKE_CXX_STANDARD 17)

# Library
add_library(math_lib src/math_utils.cpp)

# Publicly expose the include directory so consumers can see headers
target_include_directories(math_lib PUBLIC include)

# Executable
add_executable(app src/main.cpp)
target_link_libraries(app PRIVATE math_lib)
