cmake_minimum_required(VERSION 3.15)
project(ex11_unit_testing)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(GTest REQUIRED)

include_directories(include)

# Main App
add_executable(main src/main.cpp)

# Tests
enable_testing()
add_executable(run_tests tests/test_math.cpp)
target_link_libraries(run_tests GTest::gtest_main)
add_test(NAME UnitTests COMMAND run_tests)
