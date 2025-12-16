cmake_minimum_required(VERSION 3.15)
project(ex08_postprocessing_answer CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_executable(postprocess src/main.cpp)
target_link_libraries(postprocess PRIVATE OpenCV::opencv)

# Testing
enable_testing()
add_executable(test_post tests/test_post.cpp)
target_link_libraries(test_post PRIVATE OpenCV::opencv)
add_test(NAME PostProcessingTest COMMAND test_post)
