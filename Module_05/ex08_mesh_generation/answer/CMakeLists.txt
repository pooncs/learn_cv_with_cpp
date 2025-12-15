cmake_minimum_required(VERSION 3.15)
project(ex08_mesh_generation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(fmt REQUIRED)
find_package(GTest REQUIRED)

include_directories(include)

add_library(mesh src/mesh.cpp)
target_link_libraries(mesh fmt::fmt)

add_executable(main src/main.cpp)
target_link_libraries(main mesh)

enable_testing()
add_executable(run_tests tests/test_mesh.cpp)
target_link_libraries(run_tests mesh GTest::gtest_main)
add_test(NAME UnitTests COMMAND run_tests)
