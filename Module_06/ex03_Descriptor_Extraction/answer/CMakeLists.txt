cmake_minimum_required(VERSION 3.15)
project(ex03_orb_answer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(include)

add_executable(orb_demo src/main.cpp src/orb_custom.cpp)
target_link_libraries(orb_demo PRIVATE ${OpenCV_LIBS})
