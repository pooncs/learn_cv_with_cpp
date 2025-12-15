cmake_minimum_required(VERSION 3.15)
project(ex08_pnp_solver)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)
find_package(GTest REQUIRED)

include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(pnp src/pnp.cpp)
target_link_libraries(pnp ${OpenCV_LIBS} fmt::fmt)

add_executable(main src/main.cpp)
target_link_libraries(main pnp)

enable_testing()
add_executable(run_tests tests/test_pnp.cpp)
target_link_libraries(run_tests pnp GTest::gtest_main)
add_test(NAME UnitTests COMMAND run_tests)
