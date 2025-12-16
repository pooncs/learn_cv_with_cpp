cmake_minimum_required(VERSION 3.10)
project(g2o_curve_fitting_todo)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 REQUIRED)
find_package(g2o QUIET)

add_executable(curve_fitting_todo src/main.cpp)

if (g2o_FOUND)
    target_link_libraries(curve_fitting_todo PRIVATE g2o::core g2o::stuff g2o::solver_dense Eigen3::Eigen)
endif()
