# Exercise 04: Linking and Propagation

## Goal
Understand how usage requirements (`PUBLIC`, `PRIVATE`, `INTERFACE`) propagate dependencies.

## Learning Objectives
1.  **PRIVATE:** Implementation detail. Not visible to consumers.
2.  **INTERFACE:** Usage requirement. Not used by implementation, but needed by consumers (e.g., header-only libs).
3.  **PUBLIC:** Used by implementation AND consumers.

## Practical Motivation
If LibA uses Boost internally but doesn't expose Boost types in its header, it should link Boost as `PRIVATE`. If it returns a Boost object, it must be `PUBLIC`.

## Step-by-Step Instructions

### Task 1: Create LibA (Interface)
1.  `include/lib_a.hpp`: Defines a constant `VAL_A = 10`.
2.  `CMakeLists.txt`: `add_library(lib_a INTERFACE)`. `target_include_directories(lib_a INTERFACE include)`.

### Task 2: Create LibB (Static)
1.  `include/lib_b.hpp`: `int getValB();`.
2.  `src/lib_b.cpp`: Includes `lib_a.hpp` and returns `VAL_A * 2`.
3.  `CMakeLists.txt`: `add_library(lib_b ...)`. Link `lib_a` as `PRIVATE` (because `lib_b.hpp` doesn't include `lib_a.hpp`).

### Task 3: Create App
1.  `src/main.cpp`: Calls `getValB()`. Does NOT include `lib_a.hpp`.
2.  `CMakeLists.txt`: Link `lib_b`.

## Common Pitfalls
1.  **Over-linking:** Making everything PUBLIC pollutes the namespace and increases build times.
2.  **Under-linking:** If you use a type in a header but link PRIVATE, consumers will fail to compile.

## Verification
1.  Build and run.
2.  Try including `lib_a.hpp` in `main.cpp`. It should fail if linked PRIVATE (and headers are separate). Wait, if `lib_b` is in the same project and we use `target_include_directories` with absolute paths or relative to source, it might still be visible if folders overlap. But strictly speaking, the include path for `lib_a` is not propagated.
