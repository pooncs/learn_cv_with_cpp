cmake_minimum_required(VERSION 3.15)
project(ex11_pool_answer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(include)

add_executable(pool_demo src/main.cpp src/buffer_pool.cpp)
target_link_libraries(pool_demo PRIVATE ${OpenCV_LIBS})
