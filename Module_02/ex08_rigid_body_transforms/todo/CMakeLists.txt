cmake_minimum_required(VERSION 3.15)
project(ex08_rigid_body_transforms)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)

include_directories(include)

if(TARGET Eigen3::Eigen)
    get_target_property(EIGEN_INC Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
    include_directories(${EIGEN_INC})
endif()

add_library(se3 src/se3.cpp)
if(TARGET Eigen3::Eigen)
    target_link_libraries(se3 Eigen3::Eigen)
endif()

add_executable(main src/main.cpp)
target_link_libraries(main se3)

enable_testing()
add_executable(run_tests tests/test_se3.cpp)
target_link_libraries(run_tests se3)
add_test(NAME UnitTests COMMAND run_tests)
