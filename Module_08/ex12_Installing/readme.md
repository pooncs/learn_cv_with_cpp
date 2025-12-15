# Exercise 12: Installing

## Goal
Configure the project to be installed (copied) to a system directory or a local folder.

## Learning Objectives
1.  **Install Targets:** `install(TARGETS my_app DESTINATION bin)`.
2.  **Install Files:** `install(FILES header.hpp DESTINATION include)`.
3.  **Prefix:** `CMAKE_INSTALL_PREFIX` controls where files go (default `/usr/local` on Linux, `Program Files` on Windows).

## Practical Motivation
After building a library, you want to distribute it so others can use it. `make install` does this.

## Step-by-Step Instructions

### Task 1: main.cpp
A simple "Hello Install" program.

### Task 2: CMakeLists.txt
1.  `add_executable(my_app ...)`.
2.  `install(TARGETS my_app DESTINATION bin)`.
3.  `install(FILES src/readme.txt DESTINATION share/my_app)`.

### Task 3: Install
```bash
cmake -DCMAKE_INSTALL_PREFIX=./install_dir ..
cmake --build . --target install
```

## Verification
Check that `install_dir/bin/my_app` exists.
