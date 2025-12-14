find_package(GTest REQUIRED)

add_executable(unit_tests test_memory.cpp)
target_link_libraries(unit_tests PRIVATE GTest::gtest_main CUDA::cudart)

include(GoogleTest)
gtest_discover_tests(unit_tests)
