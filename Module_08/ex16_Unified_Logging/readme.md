# Module 08 - Exercise 16: Unified Logging Interface

## Goal
Create a logging facade that can switch backends (spdlog vs glog) at compile time.

## Concept
Large projects often abstract logging to avoid vendor lock-in.
We will define macros `LOG_INFO`, `LOG_WARN` that map to either library based on a CMake definition.

## Task
1.  Define `core/logger.hpp`.
2.  Use `#ifdef USE_SPDLOG` to call `spdlog::info`.
3.  Use `#ifdef USE_GLOG` to call `LOG(INFO)`.
4.  Configure CMake to select the backend.

## Instructions
1.  Navigate to `todo/`.
2.  Implement `logger.hpp`.
3.  Implement `main.cpp` using `LOG_INFO(...)`.
4.  In `CMakeLists.txt`, set `USE_SPDLOG` or `USE_GLOG`.

## Build
```bash
mkdir build
cd build
conan install .. --build=missing
cmake .. --preset conan-default
cmake --build . --config Release
./bin/unified_log
```
