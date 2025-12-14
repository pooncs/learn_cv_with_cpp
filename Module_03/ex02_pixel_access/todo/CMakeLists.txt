cmake_minimum_required(VERSION 3.15)
project(ex02_pixel_access_todo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(pixel_access src/pixel_access.cpp)
target_link_libraries(pixel_access ${OpenCV_LIBS})

add_executable(main src/main.cpp)
target_link_libraries(main pixel_access ${OpenCV_LIBS})

enable_testing()
add_executable(run_tests tests/test_pixel_access.cpp)
target_link_libraries(run_tests pixel_access ${OpenCV_LIBS})
add_test(NAME UnitTests COMMAND run_tests)
