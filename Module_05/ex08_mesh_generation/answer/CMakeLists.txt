cmake_minimum_required(VERSION 3.15)
project(ex08_mesh_generation)

set(CMAKE_CXX_STANDARD 17)

find_package(fmt REQUIRED)
find_package(GTest REQUIRED)

include_directories(include)

add_library(mesh_lib src/mesh.cpp)
target_link_libraries(mesh_lib PRIVATE fmt::fmt)

add_executable(mesh_gen src/main.cpp)
target_link_libraries(mesh_gen mesh_lib fmt::fmt)

enable_testing()
add_executable(unit_tests_bin tests/test_mesh.cpp)
target_link_libraries(unit_tests_bin mesh_lib GTest::gtest_main)
add_test(NAME UnitTests COMMAND unit_tests_bin)
