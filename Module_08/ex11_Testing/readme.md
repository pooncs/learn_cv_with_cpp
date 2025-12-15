# Exercise 11: Testing with CTest

## Goal
Use CTest to run automated tests.

## Learning Objectives
1.  **Enable Testing:** `enable_testing()`.
2.  **Add Test:** `add_test(NAME test_name COMMAND executable arg1 arg2)`.
3.  **Running:** `ctest` command.

## Practical Motivation
Running `./my_app test1`, `./my_app test2` manually is error-prone. CTest automates this.

## Step-by-Step Instructions

### Task 1: main.cpp
Write a program that takes an argument.
-   If arg is "success", return 0.
-   If arg is "fail", return 1.

### Task 2: CMakeLists.txt
1.  `add_executable(my_test src/main.cpp)`.
2.  `enable_testing()`.
3.  `add_test(NAME PassTest COMMAND my_test success)`.
4.  `add_test(NAME FailTest COMMAND my_test fail)`.
5.  Set `set_tests_properties(FailTest PROPERTIES WILL_FAIL TRUE)`.

### Task 3: Run
```bash
cmake ..
cmake --build .
ctest
```

## Common Pitfalls
1.  **enable_testing:** Must be called in the root `CMakeLists.txt` (or where you want to run ctest from).
2.  **Test Discovery:** CTest just runs commands. It doesn't know about GTest or Catch2 internals unless you use specific integration modules (like `gtest_discover_tests` in CMake 3.10+). This exercise uses simple exit-code based testing.

## Verification
`ctest` should report 100% passed (because we marked FailTest as expected to fail).
