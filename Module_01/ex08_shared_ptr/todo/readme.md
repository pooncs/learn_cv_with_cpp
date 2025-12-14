# Exercise 08: Smart Pointers II (std::shared_ptr)

## Goal
Understand **Shared Ownership** using `std::shared_ptr`. You will simulate a simple Scene Graph where multiple nodes share the same 3D Mesh data.

## Learning Objectives
1.  Understand Reference Counting (`use_count`).
2.  Use `std::make_shared<T>`.
3.  When to use `shared_ptr` vs `unique_ptr`.

## Practical Motivation
In a 3D engine or CV system:
*   You load a "Face Mesh" once (100MB).
*   You have 50 detected "Players" in the scene.
*   All 50 players should point to the **same** Mesh data.
*   The Mesh should only be deleted when the **last** player is destroyed.

This is exactly what `shared_ptr` does.

## Theory
`std::shared_ptr` maintains a control block with a reference counter.
*   Copying increments the counter.
*   Destruction decrements the counter.
*   Count == 0 -> Delete object.

## Step-by-Step Instructions

### Task 1: Create a Shared Resource
Open `src/main.cpp`.
*   Create a `std::shared_ptr<Mesh>` named `sphere_mesh` using `std::make_shared`.

### Task 2: Share it
*   Create a `std::vector<Node>`.
*   Add multiple Nodes, passing `sphere_mesh` to each.
*   Print `sphere_mesh.use_count()` to see it increase.

### Task 3: Observe Destruction
*   Clear the vector. The nodes are destroyed.
*   Check `use_count` again. It should drop back to 1 (because `sphere_mesh` local variable still holds it).
*   When `main` ends, the mesh is finally deleted.

## Common Pitfalls
*   **Cyclic References:** Node A holds shared_ptr to Node B. Node B holds shared_ptr to Node A. Count never reaches 0. Memory Leak!
    *   **Solution:** Use `std::weak_ptr` for back-references.
*   **Performance:** `shared_ptr` is slower than `unique_ptr` due to atomic counter increments. Don't use it for *everything*.

## Verification
Compile and run.
```bash
cd todo
mkdir build && cd build
cmake ..
cmake --build .
./main
```
Watch the console logs for creation/destruction messages.
