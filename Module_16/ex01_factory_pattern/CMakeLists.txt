cmake_minimum_required(VERSION 3.15)
project(Module16_Ex01_Factory)

find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)

include_directories(answer/include)

add_executable(ex01_factory_answer answer/src/main.cpp)
target_link_libraries(ex01_factory_answer PRIVATE opencv_core opencv_imgproc opencv_highgui fmt::fmt)
target_compile_features(ex01_factory_answer PRIVATE cxx_std_17)

add_executable(ex01_factory_todo todo/src/main.cpp)
target_include_directories(ex01_factory_todo PRIVATE todo/include)
target_link_libraries(ex01_factory_todo PRIVATE opencv_core opencv_imgproc opencv_highgui fmt::fmt)
target_compile_features(ex01_factory_todo PRIVATE cxx_std_17)
