cmake_minimum_required(VERSION 3.15)
project(ex16_unified_log)

set(CMAKE_CXX_STANDARD 17)

# Select Backend
option(USE_SPDLOG "Use spdlog backend" ON)
option(USE_GLOG "Use glog backend" OFF)

find_package(fmt REQUIRED)

if(USE_SPDLOG)
    find_package(spdlog REQUIRED)
    add_compile_definitions(USE_SPDLOG)
elseif(USE_GLOG)
    find_package(glog REQUIRED)
    add_compile_definitions(USE_GLOG)
endif()

add_executable(unified_log src/main.cpp)
target_link_libraries(unified_log PRIVATE fmt::fmt)

if(USE_SPDLOG)
    target_link_libraries(unified_log PRIVATE spdlog::spdlog)
elseif(USE_GLOG)
    target_link_libraries(unified_log PRIVATE glog::glog)
endif()
