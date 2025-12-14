# Exercise 11: Unit Testing Basics

## Goal
Write unit tests for a simple math utility class using GoogleTest.

## Learning Objectives
1.  Understand the structure of a Unit Test (Arrange, Act, Assert).
2.  Use GoogleTest macros (`EXPECT_EQ`, `ASSERT_TRUE`, etc.).
3.  Integrate testing into the build system (CMake + CTest).

## Practical Motivation
Testing is non-negotiable in production CV code. 
- **Regression Testing**: Ensure new features don't break old ones.
- **Documentation**: Tests describe how the code is supposed to work.
- **Confidence**: Refactor code without fear.

## Theory & Background

### GoogleTest
- `TEST(TestSuiteName, TestName) { ... }`: Defines a test.
- `EXPECT_EQ(val1, val2)`: Non-fatal failure.
- `ASSERT_EQ(val1, val2)`: Fatal failure (stops test).
- `TEST_F(FixtureClass, TestName)`: Test with setup/teardown.

### Math Utility
We will test a simple `MathUtils` class that performs:
- Factorial
- Fibonacci
- Linear Interpolation (Lerp)

## Implementation Tasks

### Task 1: Implement MathUtils
Implement `factorial(n)`, `fibonacci(n)`, and `lerp(a, b, t)`.

### Task 2: Write Tests
Write tests covering:
- Base cases (e.g., factorial(0)).
- Normal cases.
- Edge cases (if any).

## Common Pitfalls
- Writing tests that depend on each other (tests should be independent).
- Testing implementation details instead of public behavior.
