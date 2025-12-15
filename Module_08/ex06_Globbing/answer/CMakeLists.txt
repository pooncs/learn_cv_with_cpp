cmake_minimum_required(VERSION 3.10)
project(ex06_glob LANGUAGES CXX)

# Collect all .cpp files in src/
file(GLOB SRC_FILES "src/*.cpp")
# Or recursively: file(GLOB_RECURSE SRC_FILES "src/*.cpp")

message(STATUS "Found sources: ${SRC_FILES}")

add_executable(my_app ${SRC_FILES})
