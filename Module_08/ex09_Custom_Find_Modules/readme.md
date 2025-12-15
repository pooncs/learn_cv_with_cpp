# Exercise 09: Custom Find Modules

## Goal
Learn how to write a custom `FindXXX.cmake` module to locate a library that doesn't provide a CMake config file.

## Learning Objectives
1.  **CMAKE_MODULE_PATH:** Where CMake looks for `FindXXX.cmake`.
2.  **find_path:** Locate header files.
3.  **find_library:** Locate `.lib` or `.a` files.
4.  **Imported Targets:** Create `MyLib::MyLib` target with properties.

## Practical Motivation
You downloaded a C library "Foo" that only has a Makefile. You want to use `find_package(Foo)` in your project.

## Step-by-Step Instructions

### Task 1: Fake Library
Create a folder `fake_lib` with `include/foo.h` and `lib/foo.lib` (empty files are fine for this demo, or minimal content).

### Task 2: Write cmake/FindFoo.cmake
1.  `find_path(FOO_INCLUDE_DIR NAMES foo.h PATHS ...)`
2.  `find_library(FOO_LIBRARY NAMES foo PATHS ...)`
3.  `include(FindPackageHandleStandardArgs)`
4.  `find_package_handle_standard_args(Foo ...)`
5.  Create `add_library(Foo::Foo IMPORTED)` and set properties.

### Task 3: CMakeLists.txt
1.  Set `CMAKE_MODULE_PATH`.
2.  `find_package(Foo REQUIRED)`.
3.  Link `Foo::Foo`.

## Common Pitfalls
1.  **Hardcoded Paths:** Use `PATHS` or `HINTS` but don't hardcode absolute paths if possible (use env vars).
2.  **Case:** `FindFoo.cmake` matches `find_package(Foo)`.

## Verification
CMake configure should succeed and print "Found Foo".
