cmake_minimum_required(VERSION 3.10)
project(ex08_findpkg LANGUAGES CXX)

find_package(OpenCV REQUIRED)

add_executable(my_app src/main.cpp)

# Link against OpenCV
target_link_libraries(my_app PRIVATE ${OpenCV_LIBS})
