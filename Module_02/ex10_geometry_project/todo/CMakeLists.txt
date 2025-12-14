cmake_minimum_required(VERSION 3.15)
project(ex10_geometry_project_todo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)

include_directories(include)
if(TARGET Eigen3::Eigen)
    get_target_property(EIGEN_INC Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
    include_directories(${EIGEN_INC})
endif()

add_library(transformer src/transformer.cpp)
if(TARGET Eigen3::Eigen)
    target_link_libraries(transformer Eigen3::Eigen)
endif()

add_executable(main src/main.cpp)
target_link_libraries(main transformer)

enable_testing()
add_executable(run_tests tests/test_transformer.cpp)
target_link_libraries(run_tests transformer)
add_test(NAME UnitTests COMMAND run_tests)
