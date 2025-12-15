cmake_minimum_required(VERSION 3.10)
project(ex13_package LANGUAGES CXX)

add_executable(my_app src/main.cpp)

install(TARGETS my_app DESTINATION bin)

# Packaging Setup
set(CPACK_PACKAGE_NAME "MyCoolApp")
set(CPACK_PACKAGE_VENDOR "LearnCV")
set(CPACK_PACKAGE_VERSION "1.0.0")

# Choose generator (ZIP is portable)
set(CPACK_GENERATOR "ZIP")

include(CPack)
