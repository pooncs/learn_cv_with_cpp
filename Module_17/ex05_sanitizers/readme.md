# Exercise 05: Runtime Sanitizers (ASan/UBSan)

## Goal
Detect memory errors (leaks, buffer overflows) and undefined behavior at runtime using AddressSanitizer (ASan) and UndefinedBehaviorSanitizer (UBSan).

## Learning Objectives
1.  **Memory Safety:** Understand common C++ memory errors (use-after-free, out-of-bounds).
2.  **Sanitizers:** Learn how ASan and UBSan instrument code to catch these errors.
3.  **CMake Configuration:** Enable sanitizers in the build system.
4.  **Debugging:** Interpret sanitizer error reports.

## Practical Motivation
C++ allows you to shoot yourself in the foot. A buffer overflow might not crash immediately but corrupts data silently, causing a crash hours later. ASan catches these errors *instantly* at the point of occurrence.

## Theory: Instrumentation
Sanitizers add runtime checks to every memory access and arithmetic operation.
-   **ASan:** Shadow memory to track allocation states.
-   **UBSan:** Checks for integer overflow, null pointer dereference, etc.

## Step-by-Step Instructions

### Task 1: Enable Sanitizers
In `CMakeLists.txt`:
```cmake
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
elseif(MSVC)
    add_compile_options(/fsanitize=address)
endif()
```
*Note: MSVC support for ASan is relatively new but available.*

### Task 2: Write Buggy Code
In `src/buggy.cpp`:
```cpp
int* p = new int[10];
p[10] = 0; // Heap buffer overflow
delete[] p;
// delete[] p; // Double free
```

### Task 3: Run and Analyze
Build and run the executable. It should crash with a colorful report detailing the memory error.

## Code Hints
-   **MSVC Setup:** You might need to install the ASan component in Visual Studio Installer.
-   **Performance:** Sanitizers slow down execution (2x-5x). Use them in Debug or RelWithDebInfo builds, not for performance benchmarking.

## Verification
The program must produce an ASan report and exit with a non-zero code.
