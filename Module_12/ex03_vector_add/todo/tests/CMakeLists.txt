find_package(GTest REQUIRED)

# Link the kernel wrapper implementation from src
# Since it's not a library, we might need to compile vector_add.cu again or add it to sources
add_executable(unit_tests test_vector_add.cpp ../src/vector_add.cu)
target_include_directories(unit_tests PRIVATE ../include)
target_link_libraries(unit_tests PRIVATE GTest::gtest_main CUDA::cudart)

include(GoogleTest)
gtest_discover_tests(unit_tests)
