cmake_minimum_required(VERSION 3.15)
project(ex10_custom_filter_todo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(custom_filter src/custom_filter.cpp)
target_link_libraries(custom_filter ${OpenCV_LIBS})

add_executable(main src/main.cpp)
target_link_libraries(main custom_filter ${OpenCV_LIBS})

enable_testing()
add_executable(run_tests tests/test_custom_filter.cpp)
target_link_libraries(run_tests custom_filter ${OpenCV_LIBS})
add_test(NAME UnitTests COMMAND run_tests)
