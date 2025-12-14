# Exercise 07: Smart Pointers I (std::unique_ptr)

## Goal
Understand **Exclusive Ownership** using `std::unique_ptr`. You will implement a factory pattern that creates sensors (Camera/Lidar) without manual memory management.

## Learning Objectives
1.  Understand `std::unique_ptr<T>` vs raw pointers.
2.  Use `std::make_unique<T>` (C++14/17 preferred).
3.  Polymorphism with smart pointers.

## Practical Motivation
In older C++ CV libraries, you see `cv::Ptr<FeatureDetector>`. This is basically a smart pointer.
Using `unique_ptr` ensures that:
1.  **No Memory Leaks:** If the function returns early (or throws), the object is deleted.
2.  **Clear Ownership:** "I own this sensor driver. No one else deletes it."

## Theory
`std::unique_ptr` cannot be copied, only **moved**.
```cpp
auto p1 = std::make_unique<int>(5);
auto p2 = p1; // ERROR: Copy deleted
auto p3 = std::move(p1); // OK: p1 is now null
```

## Step-by-Step Instructions

### Task 1: The Factory Function
Open `src/main.cpp`.
*   Implement `createSensor(type)` which returns `std::unique_ptr<Sensor>`.
*   Use `std::make_unique<Camera>()` or `std::make_unique<Lidar>()`.

### Task 2: Using the Pointer
In `main()`:
1.  Call the factory: `auto sensor = createSensor("camera");`.
2.  Call a method: `sensor->read();`.
3.  **Observation:** Do not call `delete`. Observe the console output to confirm the destructor is called automatically when `sensor` goes out of scope.

## Common Pitfalls
*   **Passing to Functions:**
    *   `void func(std::unique_ptr<T> p)` -> Takes ownership (you must `std::move` into it).
    *   `void func(const std::unique_ptr<T>& p)` -> Weird. Just pass `T*` or `T&` if you don't care about ownership.
    *   **Best Practice:** Use `T*` for non-owning access. `func(sensor.get())`.

## Verification
Compile and run.
```bash
cd todo
mkdir build && cd build
cmake ..
cmake --build .
./main
```
Output must show "Sensor destroyed" automatically.
