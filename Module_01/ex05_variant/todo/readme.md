# Exercise 05: Type-Safe Unions (std::variant)

## Goal
Implement polymorphism without inheritance (and vtables) using **`std::variant`**. You will build a shape container that can hold `Circle` or `Rectangle` objects safely.

## Learning Objectives
1.  Understand `std::variant` as a type-safe `union`.
2.  Use `std::visit` to apply a function (visitor) to the stored value.
3.  Check active types using `std::holds_alternative` or `std::get_if`.

## Practical Motivation
In a CV pipeline, a "Detection" might be:
1.  A **Bounding Box** (Rectangle).
2.  A **Mask** (Polygon).
3.  A **Keypoint** (Point).

Using inheritance (`class DetectionBase`) requires heap allocation (`std::vector<DetectionBase*>`) and virtual function calls, which ruin cache locality.
`std::variant<Box, Mask, Keypoint>` stores the data **inline** in the vector. It's faster, cache-friendly, and safer.

## Theory
`std::variant<A, B>` creates a storage block large enough to hold `max(sizeof(A), sizeof(B))`. It knows which type is currently active.
*   **Visitor Pattern:** `std::visit` takes a generic lambda or functor and applies it to the current value.

## Step-by-Step Instructions

### Task 1: Define the Variant
Open `src/main.cpp`.
*   Define `using Shape = std::variant<Circle, Rectangle>;`.

### Task 2: Create the Visitor
Define a struct `AreaVisitor` (or use a generic lambda) that overloads `operator()`:
```cpp
struct AreaVisitor {
    double operator()(const Circle& c) { return ...; }
    double operator()(const Rectangle& r) { return ...; }
};
```

### Task 3: Process a List of Shapes
1.  Create a `std::vector<Shape>`.
2.  Push a `Circle` and a `Rectangle` into it.
3.  Loop through the vector and calculate the area using `std::visit(AreaVisitor{}, shape)`.

### Task 4: Type Checking
Inside the loop, check specifically if the shape is a Circle using `std::get_if<Circle>(&shape)`. If it is, print its radius.

## Common Pitfalls
*   **Empty Variants:** Unlike `std::optional`, a `std::variant` is rarely "empty" (unless in valid state). It must hold one of the types.
*   **Visitor Exhaustiveness:** Your visitor must handle *all* possible types in the variant, or it won't compile.

## Verification
Compile and run.
```bash
cd todo
mkdir build && cd build
cmake ..
cmake --build .
./main
```
Output should show areas for both circles and rectangles.
