cmake_minimum_required(VERSION 3.15)
project(ex01_ls_todo CXX)

find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)
find_package(GTest REQUIRED)

add_executable(main src/main.cpp)
target_include_directories(main PRIVATE include)
target_link_libraries(main PRIVATE Eigen3::Eigen fmt::fmt)
target_compile_features(main PRIVATE cxx_std_17)

enable_testing()
add_executable(test_ls tests/test_ls.cpp)
target_include_directories(test_ls PRIVATE include)
target_link_libraries(test_ls PRIVATE Eigen3::Eigen GTest::gtest GTest::gtest_main)
add_test(NAME LSTest COMMAND test_ls)
