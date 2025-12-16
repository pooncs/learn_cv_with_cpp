cmake_minimum_required(VERSION 3.10)
project(g2o_curve_fitting)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 REQUIRED)
# Note: g2o usually requires manual path setting or local build if not installed system-wide
# We assume it's available or managed by Conan
find_package(g2o QUIET)

if (g2o_FOUND OR EXISTS ${g2o_DIR})
    add_executable(curve_fitting src/main.cpp)
    target_link_libraries(curve_fitting PRIVATE g2o::core g2o::stuff g2o::solver_dense Eigen3::Eigen)
else()
    message(STATUS "g2o not found. Skipping build.")
    add_executable(curve_fitting src/main.cpp) # Placeholder
endif()
