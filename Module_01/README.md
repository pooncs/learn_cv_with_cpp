# Module 01: Modern C++17 Core for CV

## Exercises
1.  **ex01_auto**: Refactor legacy code using `auto` and `decltype`.
2.  **ex02_stl_perf**: Benchmark `std::vector` vs `std::list`.
3.  **ex03_structured_bindings**: Unpack tuples and structs.
4.  **ex04_error_handling**: Use `tl::expected` (modern error handling).
5.  **ex05_variant**: Implement a shape container with `std::variant`.
6.  **ex06_lambdas**: Advanced sorting and filtering.
7.  **ex07_unique_ptr**: Factory functions and ownership.
8.  **ex08_shared_ptr**: Scene graph simulation.
9.  **ex09_move_semantics**: Optimize Matrix class.
10. **ex10_raii_wrapper**: The `ImageBuffer` class (Rule of Five).
11. **ex11_unit_testing**: Testing with GTest.
12. **ex12_scoped_timer**: RAII Timer for profiling.
13. **ex13_minimal_logger**: Simple logger class.
14. **ex14_file_io**: Reading/Writing files.

## Build Instructions
Each exercise has a `todo` and `answer` directory.
Use Conan 2.0 + CMake to build.

```bash
cd exXX_.../todo
conan install . --build=missing
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build .
ctest
```
