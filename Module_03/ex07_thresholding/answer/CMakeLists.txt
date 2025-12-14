cmake_minimum_required(VERSION 3.15)
project(ex07_thresholding)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(thresholding src/thresholding.cpp)
target_link_libraries(thresholding ${OpenCV_LIBS})

add_executable(main src/main.cpp)
target_link_libraries(main thresholding ${OpenCV_LIBS})
