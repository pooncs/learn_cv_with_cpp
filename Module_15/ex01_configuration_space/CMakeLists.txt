cmake_minimum_required(VERSION 3.15)
project(Module15_Ex01_CSpace)

find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)

add_executable(ex01_cspace_answer answer/src/main.cpp)
target_link_libraries(ex01_cspace_answer PRIVATE opencv_core opencv_imgproc opencv_highgui fmt::fmt)
target_compile_features(ex01_cspace_answer PRIVATE cxx_std_17)

add_executable(ex01_cspace_todo todo/src/main.cpp)
target_link_libraries(ex01_cspace_todo PRIVATE opencv_core opencv_imgproc opencv_highgui fmt::fmt)
target_compile_features(ex01_cspace_todo PRIVATE cxx_std_17)
