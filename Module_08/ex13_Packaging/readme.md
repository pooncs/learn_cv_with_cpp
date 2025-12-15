# Exercise 13: Packaging with CPack

## Goal
Create an installer or a zip archive of your project using CPack.

## Learning Objectives
1.  **CPack:** The packaging tool bundled with CMake.
2.  **Configuration:** `set(CPACK_GENERATOR "ZIP")` or "NSIS" (Windows Installer).
3.  **Inclusion:** `include(CPack)` at the *end* of your root `CMakeLists.txt`.

## Practical Motivation
Users prefer installers (MSI, DEB, RPM) over running `cmake` and `make install`.

## Step-by-Step Instructions

### Task 1: main.cpp
A simple app.

### Task 2: CMakeLists.txt
1.  `install(TARGETS ...)` (Prerequisite: CPack packages what is installed).
2.  Set `CPACK_PACKAGE_NAME`, `CPACK_PACKAGE_VERSION`.
3.  `include(CPack)`.

### Task 3: Package
```bash
cmake ..
cmake --build . --config Release
cpack
```

## Common Pitfalls
1.  **Missing Install:** If you don't `install()` anything, the package will be empty.
2.  **Order:** `include(CPack)` must be last.

## Verification
Check for `MyPackage-1.0.0-win64.zip` (or similar) in the build directory.
