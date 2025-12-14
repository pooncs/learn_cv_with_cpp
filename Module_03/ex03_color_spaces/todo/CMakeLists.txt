cmake_minimum_required(VERSION 3.15)
project(ex03_color_spaces_todo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(color_spaces src/color_spaces.cpp)
target_link_libraries(color_spaces ${OpenCV_LIBS})

add_executable(main src/main.cpp)
target_link_libraries(main color_spaces ${OpenCV_LIBS})

enable_testing()
add_executable(run_tests tests/test_color_spaces.cpp)
target_link_libraries(run_tests color_spaces ${OpenCV_LIBS})
add_test(NAME UnitTests COMMAND run_tests)
