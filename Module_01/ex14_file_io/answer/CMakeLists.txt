cmake_minimum_required(VERSION 3.15)
project(ex14_file_io)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(nlohmann_json REQUIRED)

include_directories(include)

add_library(fileio src/FileIO.cpp)
target_link_libraries(fileio nlohmann_json::nlohmann_json)

add_executable(main src/main.cpp)
target_link_libraries(main fileio)
