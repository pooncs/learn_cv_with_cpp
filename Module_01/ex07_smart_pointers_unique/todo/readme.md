# Exercise 07: Smart Pointers I (Unique Pointers)

## Goal
Master `std::unique_ptr` for exclusive resource ownership. You will implement a factory function that returns a sensor driver, ensuring it is automatically cleaned up.

## Learning Objectives
1.  Understand **Exclusive Ownership**: Only one pointer owns the resource.
2.  Use `std::make_unique` to create objects.
3.  Transfer ownership using `std::move`.
4.  Understand why `unique_ptr` has zero overhead compared to raw pointers.

## Analogy: The Exclusive Key
*   **Old C++ (Raw Pointers):** You copy the key to the "Sensor Room" and give it to 5 people.
    *   *Risk:* Who locks the door? If Person A locks it (deletes) and Person B tries to enter... crash. If nobody locks it... memory leak.
*   **Modern C++ (`std::unique_ptr`):** There is **only one physical key**.
    *   You can hold it.
    *   You can **give** it to someone else (`std::move`), but then you don't have it anymore.
    *   When the key holder leaves the building (scope ends), the door is automatically locked and the room cleaned.

## Practical Motivation
In CV, we manage heavy resources:
*   Camera Drivers (OpenCV VideoCapture).
*   GPU Memory (CudaMalloc).
*   File Handles.

Using `unique_ptr` ensures that if your function throws an exception or returns early, the resource is closed properly. No more `delete` in `catch` blocks!

## Step-by-Step Instructions

### Task 1: Create a Sensor Class
Open `src/main.cpp`. Define a dummy class `Sensor` that prints "Sensor Open" in constructor and "Sensor Closed" in destructor.

### Task 2: Factory Function
Implement `create_sensor()` that returns `std::unique_ptr<Sensor>`.
*   Use `std::make_unique<Sensor>()`.

### Task 3: Ownership Transfer
In `main()`:
1.  Create a sensor: `auto my_sensor = create_sensor();`.
2.  Try to copy it: `auto s2 = my_sensor;` (This should fail compilation!).
3.  Move it: `auto s2 = std::move(my_sensor);`.
4.  Verify that `my_sensor` is now empty (nullptr).

### Task 4: Passing to Functions
Write a function `process_sensor(std::unique_ptr<Sensor> s)`.
*   Call it by moving your sensor into it.
*   Observe when the destructor is called (it should be when `process_sensor` finishes).

## Verification
Compile and run.
```bash
cd todo
mkdir build && cd build
cmake ..
cmake --build .
./main
```
Output should show "Sensor Open", then "Sensor Closed" at the correct times.
