# Exercise 08: Finding Packages

## Goal
Use `find_package` to locate external libraries installed on the system (or via Conan/Vcpkg) and link against them.

## Learning Objectives
1.  **Config Mode vs Module Mode:** `FindPackage.cmake` vs `PackageConfig.cmake`.
2.  **Imported Targets:** Linking against `OpenCV::OpenCV` instead of `${OpenCV_LIBS}` (modern way).
3.  **REQUIRED:** Fails if not found.

## Practical Motivation
You don't want to hardcode paths to libraries. `find_package` abstracts the search process.

## Step-by-Step Instructions

### Task 1: main.cpp
Write a program that uses OpenCV to create a 100x100 black image and print its size.

### Task 2: CMakeLists.txt
1.  `find_package(OpenCV REQUIRED)`.
2.  `add_executable(...)`.
3.  `target_link_libraries(my_app PRIVATE ${OpenCV_LIBS})` (Legacy) OR `OpenCV::OpenCV` (Modern, if available).

### Task 3: Conan Integration
Ensure you run `conan install` so that `find_package` can find OpenCV.

## Common Pitfalls
1.  **Case Sensitivity:** `find_package(OpenCV)` vs `find_package(opencv)`. Usually matches the package name.
2.  **Variables:** Older packages set `Package_INCLUDE_DIRS` and `Package_LIBRARIES`. Modern ones define targets like `Package::Package`.

## Verification
Run the executable. It should print "Image size: [100 x 100]".
