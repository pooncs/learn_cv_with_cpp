cmake_minimum_required(VERSION 3.15)
project(ex11_pool_todo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(GTest REQUIRED)

include_directories(include)

# Library
add_library(pool_lib src/buffer_pool.cpp)
target_link_libraries(pool_lib PRIVATE ${OpenCV_LIBS})

# Main Executable
add_executable(pool_app src/main.cpp)
target_link_libraries(pool_app PRIVATE pool_lib ${OpenCV_LIBS})

# Tests
enable_testing()
add_executable(pool_test tests/test_pool.cpp)
target_link_libraries(pool_test PRIVATE pool_lib GTest::gtest_main ${OpenCV_LIBS})

add_test(NAME PoolTest COMMAND pool_test)
