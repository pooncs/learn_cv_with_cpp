cmake_minimum_required(VERSION 3.15)
project(ex10_annotation_tool CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

find_package(Qt6 COMPONENTS Widgets REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(GTest REQUIRED)

add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE Qt6::Widgets nlohmann_json::nlohmann_json)

enable_testing()
add_executable(test_annotator tests/test_annotator.cpp)
target_link_libraries(test_annotator PRIVATE GTest::gtest_main Qt6::Widgets nlohmann_json::nlohmann_json)
add_test(NAME test_annotator COMMAND test_annotator)
