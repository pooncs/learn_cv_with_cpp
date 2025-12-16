# Module 08 - Exercise 14: Structured Logging with spdlog

## Goal
Implement structured and asynchronous logging using the `spdlog` library.

## Concept
`spdlog` is a very fast, header-only/compiled C++ logging library.
Features:
- **Fast**: Very low latency.
- **Async**: Logging can happen on a background thread.
- **Sinks**: Console, File, Rotating File, Daily File.

## Task
1.  Initialize a console logger (stdout) and a basic file logger.
2.  Log messages at different levels (INFO, WARN, ERROR).
3.  Demonstrate formatting (e.g., `info("User {} logged in", userId)`).

## Instructions
1.  Navigate to `todo/`.
2.  Inspect `conanfile.txt`.
3.  Edit `src/main.cpp` to use `spdlog`.

## Build
```bash
mkdir build
cd build
conan install .. --build=missing
cmake .. --preset conan-default
cmake --build . --config Release
./bin/logging_demo
```
