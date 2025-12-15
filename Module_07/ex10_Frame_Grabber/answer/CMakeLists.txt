cmake_minimum_required(VERSION 3.15)
project(ex10_grabber_answer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(include)

add_executable(grabber_demo src/main.cpp src/frame_grabber.cpp)
target_link_libraries(grabber_demo PRIVATE ${OpenCV_LIBS})
