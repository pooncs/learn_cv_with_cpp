cmake_minimum_required(VERSION 3.15)
project(ex06_histograms_todo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(histogram src/histogram.cpp)
target_link_libraries(histogram ${OpenCV_LIBS})

add_executable(main src/main.cpp)
target_link_libraries(main histogram ${OpenCV_LIBS})

enable_testing()
add_executable(run_tests tests/test_histogram.cpp)
target_link_libraries(run_tests histogram ${OpenCV_LIBS})
add_test(NAME UnitTests COMMAND run_tests)
