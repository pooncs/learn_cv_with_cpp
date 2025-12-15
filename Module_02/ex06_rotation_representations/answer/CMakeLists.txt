cmake_minimum_required(VERSION 3.15)
project(ex06_rotation_representations)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)

include_directories(include)

if(TARGET Eigen3::Eigen)
    get_target_property(EIGEN_INC Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
    include_directories(${EIGEN_INC})
endif()

add_library(rotations src/rotations.cpp)
if(TARGET Eigen3::Eigen)
    target_link_libraries(rotations Eigen3::Eigen)
endif()

add_executable(main src/main.cpp)
target_link_libraries(main rotations)
