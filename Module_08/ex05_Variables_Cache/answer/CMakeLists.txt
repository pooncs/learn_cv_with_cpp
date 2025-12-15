cmake_minimum_required(VERSION 3.10)
project(ex05_vars LANGUAGES CXX)

# Define an option (Boolean Cache Variable)
option(ENABLE_MY_FEATURE "Enable the special feature" ON)

message(STATUS "ENABLE_MY_FEATURE is set to: ${ENABLE_MY_FEATURE}")

add_executable(my_app src/main.cpp)

if(ENABLE_MY_FEATURE)
    target_compile_definitions(my_app PRIVATE MY_FEATURE)
endif()
