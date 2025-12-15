find_package(GTest REQUIRED)

add_executable(unit_tests test_npp.cpp ../src/npp_resize.cpp)
target_include_directories(unit_tests PRIVATE ../include)
target_link_libraries(unit_tests PRIVATE GTest::gtest_main CUDA::cudart CUDA::nppc CUDA::nppig CUDA::nppi)

include(GoogleTest)
gtest_discover_tests(unit_tests)
