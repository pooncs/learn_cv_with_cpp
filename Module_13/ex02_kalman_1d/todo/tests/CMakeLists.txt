find_package(GTest REQUIRED)

add_executable(unit_tests test_kalman.cpp ../src/kalman_1d.cpp)
target_include_directories(unit_tests PRIVATE ../include)
target_link_libraries(unit_tests PRIVATE GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(unit_tests)
