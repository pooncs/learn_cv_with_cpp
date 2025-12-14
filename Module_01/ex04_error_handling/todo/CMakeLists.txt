cmake_minimum_required(VERSION 3.15)
project(ex04_error_handling)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(fmt REQUIRED)
find_package(GTest REQUIRED)
find_package(tl-expected REQUIRED)

include_directories(include)

add_executable(main src/main.cpp)
target_link_libraries(main fmt::fmt tl-expected::expected)

enable_testing()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    file(GLOB TEST_SOURCES "tests/*.cpp")
    if(TEST_SOURCES)
        add_executable(run_tests ${TEST_SOURCES})
        target_link_libraries(run_tests GTest::gtest_main fmt::fmt tl-expected::expected)
        add_test(NAME UnitTests COMMAND run_tests)
    endif()
endif()
