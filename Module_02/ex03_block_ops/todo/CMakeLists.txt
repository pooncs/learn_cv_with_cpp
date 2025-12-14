cmake_minimum_required(VERSION 3.15)
project(ex03_block_ops_todo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)

include_directories(include)
if(TARGET Eigen3::Eigen)
    get_target_property(EIGEN_INC Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
    include_directories(${EIGEN_INC})
endif()

add_library(block_ops src/block_ops.cpp)
if(TARGET Eigen3::Eigen)
    target_link_libraries(block_ops Eigen3::Eigen)
endif()

add_executable(main src/main.cpp)
target_link_libraries(main block_ops)

enable_testing()
add_executable(run_tests tests/test_block_ops.cpp)
target_link_libraries(run_tests block_ops)
add_test(NAME UnitTests COMMAND run_tests)
