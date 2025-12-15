# Exercise 01: Factory Pattern for Algorithms

## Goal
Implement a `ModuleFactory` that creates instances of image processing algorithms (e.g., "EdgeDetector", "Thresholder") based on a string identifier. This allows the application to be configured dynamically (e.g., from a JSON file) without recompiling.

## Learning Objectives
1.  **Factory Method:** Understand how to decouple object creation from usage.
2.  **Registration:** Implement a mechanism to register new algorithms into the factory automatically or statically.
3.  **Polymorphism:** Use a common interface (`IAlgorithm`) for all products.
4.  **Smart Pointers:** Return `std::unique_ptr<IAlgorithm>` to manage ownership.

## Practical Motivation
Imagine a CV pipeline where the user selects "Canny" or "Sobel" from a dropdown menu. You don't want a giant `if-else` block in your main loop. A Factory Pattern allows you to say `create("Canny")` and get the correct object.

## Theory: The Factory Pattern
The factory maintains a map of `string -> CreatorFunction`.
When `create(name)` is called, it looks up the creator function and invokes it.

```cpp
using Creator = std::function<std::unique_ptr<IAlgorithm>()>;
std::map<std::string, Creator> registry;
```

## Step-by-Step Instructions

### Task 1: Define the Interface
Create `IAlgorithm` with a pure virtual method `process(const cv::Mat& input)`.

### Task 2: Implement Concrete Classes
Create `CannyDetector` and `SobelDetector` implementing `IAlgorithm`.

### Task 3: Implement the Factory
Create `AlgorithmFactory` with:
*   `register_algo(name, creator_func)`
*   `create(name)`

### Task 4: Registration
Register your algorithms in `main` (or via static initialization).

## Code Hints
*   **Lambda Creators:**
    ```cpp
    factory.register_algo("Canny", []() { return std::make_unique<CannyDetector>(); });
    ```
*   **Error Handling:** Throw an exception if the requested algorithm name is not found.

## Verification
*   Test 1: Create "Canny" and verify it returns a valid pointer.
*   Test 2: Call `process()` and check the output (console print is fine for mock).
*   Test 3: Request "Unknown" and verify it throws.
