cmake_minimum_required(VERSION 3.15)
project(Module15_Ex04_RRT)

find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)

add_executable(ex04_rrt_answer answer/src/main.cpp)
target_link_libraries(ex04_rrt_answer PRIVATE opencv_core opencv_imgproc opencv_highgui fmt::fmt)
target_compile_features(ex04_rrt_answer PRIVATE cxx_std_17)

add_executable(ex04_rrt_todo todo/src/main.cpp)
target_link_libraries(ex04_rrt_todo PRIVATE opencv_core opencv_imgproc opencv_highgui fmt::fmt)
target_compile_features(ex04_rrt_todo PRIVATE cxx_std_17)
