cmake_minimum_required(VERSION 3.15)
project(ex01_factory_todo CXX)

find_package(fmt REQUIRED)
find_package(GTest REQUIRED)

add_executable(main src/main.cpp)
target_include_directories(main PRIVATE include)
target_link_libraries(main PRIVATE fmt::fmt)
target_compile_features(main PRIVATE cxx_std_17)

enable_testing()
add_executable(test_factory tests/test_factory.cpp)
target_include_directories(test_factory PRIVATE include)
target_link_libraries(test_factory PRIVATE GTest::gtest GTest::gtest_main)
add_test(NAME FactoryTest COMMAND test_factory)
