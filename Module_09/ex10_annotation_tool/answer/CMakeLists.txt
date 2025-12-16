cmake_minimum_required(VERSION 3.15)
project(ex10_annotation_tool_answer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

find_package(Qt6 COMPONENTS Widgets REQUIRED)
find_package(nlohmann_json REQUIRED)

add_executable(annotator src/main.cpp)
target_link_libraries(annotator PRIVATE Qt6::Widgets nlohmann_json::nlohmann_json)
