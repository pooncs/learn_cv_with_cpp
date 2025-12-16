# Module 08 - Exercise 15: Google Logging (glog)

## Goal
Use Google Logging (glog) for application-level logging.

## Concept
`glog` is widely used in legacy and Google projects (like Ceres Solver).
It uses macros like `LOG(INFO) << "Message";` and `CHECK(condition)`.

## Task
1.  Initialize glog.
2.  Log INFO, WARNING, ERROR.
3.  Use `CHECK_GT(a, b)` to verify a condition (it aborts if false).

## Instructions
1.  Navigate to `todo/`.
2.  Edit `src/main.cpp`.
3.  Initialize: `google::InitGoogleLogging(argv[0]);`.
4.  Log something.

## Build
```bash
mkdir build
cd build
conan install .. --build=missing
cmake .. --preset conan-default
cmake --build . --config Release
./bin/glog_demo
```
