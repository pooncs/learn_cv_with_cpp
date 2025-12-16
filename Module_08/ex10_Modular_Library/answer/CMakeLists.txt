cmake_minimum_required(VERSION 3.10)
project(ex02_library LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Create Library
add_library(my_math src/my_math.cpp)
# PUBLIC includes means:
# 1. Used to compile my_math itself
# 2. Propagated to anyone linking to my_math
target_include_directories(my_math PUBLIC include)

# Create Executable
add_executable(my_app src/main.cpp)

# Link
target_link_libraries(my_app PRIVATE my_math)
