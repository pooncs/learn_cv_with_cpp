cmake_minimum_required(VERSION 3.15)
project(Module15_Ex03_Dijkstra)

find_package(fmt REQUIRED)

add_executable(ex03_dijkstra_answer answer/src/main.cpp)
target_link_libraries(ex03_dijkstra_answer PRIVATE fmt::fmt)
target_compile_features(ex03_dijkstra_answer PRIVATE cxx_std_17)

add_executable(ex03_dijkstra_todo todo/src/main.cpp)
target_link_libraries(ex03_dijkstra_todo PRIVATE fmt::fmt)
target_compile_features(ex03_dijkstra_todo PRIVATE cxx_std_17)
