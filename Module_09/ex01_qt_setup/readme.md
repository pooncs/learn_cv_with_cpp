# Exercise 01: Qt Setup

## Goal
Create a basic "Hello World" window using CMake and Qt6. This exercise verifies that your development environment is correctly configured to build and run Qt applications.

## Learning Objectives
1.  Configure `CMakeLists.txt` to find and link Qt6 components.
2.  Understand the basic structure of a Qt `main.cpp` (QApplication, QMainWindow).
3.  Build and run a GUI application.

## Practical Motivation
Setting up the build system is the first hurdle in GUI development. Qt requires specific CMake commands (`find_package`, `qt_add_executable`) to handle Meta-Object Compilation (MOC) and linking. Mastering this ensures a smooth development experience for complex tools.

## Theory: Qt and CMake
Qt uses a custom build step called **MOC (Meta-Object Compiler)** to handle signals and slots.
In CMake, we automate this by setting:
```cmake
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
```
We then use `find_package(Qt6 COMPONENTS Widgets REQUIRED)` to locate the libraries.

## Step-by-Step Instructions

### Task 1: Configure CMake
Open `todo/CMakeLists.txt`.
1.  Enable `CMAKE_AUTOMOC`.
2.  Find the `Qt6` package with `Widgets` component.
3.  Link the executable against `Qt6::Widgets`.

### Task 2: Implement Main Window
Open `todo/src/main.cpp`.
1.  Include `<QApplication>` and `<QMainWindow>`.
2.  Create a `QApplication` object.
3.  Create a `QMainWindow`.
4.  Set the window title to "Hello Qt".
5.  Show the window.
6.  Return `app.exec()`.

## Common Pitfalls
*   **Missing AUTOMOC:** If you get linker errors about `vtable`, you likely forgot `set(CMAKE_AUTOMOC ON)`.
*   **Runtime DLL missing:** On Windows, ensure the Qt DLLs are in your PATH or copied to the build directory (Conan usually handles this environment setup).

## Verification
Build and run the application. A blank window titled "Hello Qt" should appear.
