cmake_minimum_required(VERSION 3.15)
project(ex09_least_squares_todo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)

include_directories(include)
if(TARGET Eigen3::Eigen)
    get_target_property(EIGEN_INC Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
    include_directories(${EIGEN_INC})
endif()

add_library(least_squares src/least_squares.cpp)
if(TARGET Eigen3::Eigen)
    target_link_libraries(least_squares Eigen3::Eigen)
endif()

add_executable(main src/main.cpp)
target_link_libraries(main least_squares)

enable_testing()
add_executable(run_tests tests/test_least_squares.cpp)
target_link_libraries(run_tests least_squares)
add_test(NAME UnitTests COMMAND run_tests)
