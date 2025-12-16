# Exercise 11: Unit Testing Basics

## Goal
Write unit tests for a simple math utility class using GoogleTest.

## Learning Objectives
1.  Understand the structure of a Unit Test (Arrange, Act, Assert).
2.  Use GoogleTest macros (`EXPECT_EQ`, `ASSERT_TRUE`, etc.).
3.  Integrate testing into the build system (CMake + CTest).

## Analogy: The Robot Safety Inspectors
*   **Old C++ (Manual/Printf Testing):** You build a car. You drive it into a wall to see if the airbag works.
    *   *Problem:* The car is destroyed. You have to rebuild it to test the brakes.
*   **Modern C++ (Unit Testing):** You have a team of **Indestructible Robot Inspectors**.
    *   Every time you change a screw, the robots instantly check every part of the car (Engine, Brakes, Airbag) in parallel.
    *   If a robot finds a flaw, it beeps loudly ("Test Failed: Expected 5, Got 4").
    *   You know exactly what broke, instantly.

## Practical Motivation
Testing is non-negotiable in production CV code. 
*   **Regression Testing**: Ensure new features don't break old ones.
*   **Documentation**: Tests describe how the code is supposed to work.
*   **Confidence**: Refactor code without fear.

## Theory & Background

### GoogleTest
*   `TEST(TestSuiteName, TestName) { ... }`: Defines a test.
*   `EXPECT_EQ(val1, val2)`: Non-fatal failure.
*   `ASSERT_EQ(val1, val2)`: Fatal failure (stops test).
*   `TEST_F(FixtureClass, TestName)`: Test with setup/teardown.

### Math Utility
We will test a simple `MathUtils` class that performs:
*   Factorial
*   Fibonacci
*   Linear Interpolation (Lerp)

## Step-by-Step Instructions

### Task 1: Implement MathUtils
Open `include/MathUtils.hpp` (or `src/MathUtils.cpp`).
Implement `factorial(n)`, `fibonacci(n)`, and `lerp(a, b, t)`.

### Task 2: Write Tests
Open `tests/test_math.cpp`. Write tests covering:
*   Base cases (e.g., factorial(0)).
*   Normal cases.
*   Edge cases (if any).

## Verification
Compile and run tests.
```bash
cd todo
mkdir build && cd build
cmake ..
cmake --build .
ctest --output-on-failure
```
Output should show all tests passing.
