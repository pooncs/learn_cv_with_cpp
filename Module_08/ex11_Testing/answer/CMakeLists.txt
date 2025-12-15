cmake_minimum_required(VERSION 3.10)
project(ex11_testing LANGUAGES CXX)

add_executable(my_test src/main.cpp)

enable_testing()

add_test(NAME PassTest COMMAND my_test success)

add_test(NAME FailTest COMMAND my_test fail)
# We expect this test to return non-zero
set_tests_properties(FailTest PROPERTIES WILL_FAIL TRUE)
