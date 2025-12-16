# Exercise 08: Smart Pointers II (Shared & Weak)

## Goal
Understand shared ownership with `std::shared_ptr` and how to break reference cycles using `std::weak_ptr`. You will simulate a Scene Graph where multiple nodes share the same Geometry data.

## Learning Objectives
1.  **Shared Ownership:** Resource is alive as long as *count > 0*.
2.  **Reference Counting:** How `shared_ptr` tracks owners.
3.  **Cyclic Dependencies:** The "Memory Leak Trap" when A points to B and B points to A.
4.  **Weak Pointers:** Observing without owning.

## Analogy: The Shared Google Doc
*   **`std::shared_ptr`:** A live link to a Google Doc.
    *   Multiple people can have the link open.
    *   The document stays "Active" as long as *at least one* person has it open.
    *   When the last person closes the tab, the document is archived/deleted.
*   **`std::weak_ptr`:** A bookmark in your browser history.
    *   It remembers where the doc was.
    *   You can try to click it (`.lock()`), but if the doc was deleted (nobody has it open), the link is dead.
    *   Crucially, your bookmark *doesn't prevent* the doc from being deleted.

## Practical Motivation
In a 3D Scene Graph:
*   You have a heavy 3D Mesh (100MB).
*   10 different "Robot" instances use this same Mesh.
*   We don't want to copy the Mesh 10 times (1GB!).
*   We use `shared_ptr<Mesh>`. All Robots point to the same data.
*   The Mesh is deleted only when *all* Robots are destroyed.

## Step-by-Step Instructions

### Task 1: The Shared Resource
Open `src/main.cpp`. Create a `Mesh` class that prints "Mesh Loaded" / "Mesh Unloaded".

### Task 2: Shared Ownership
1.  Create `std::shared_ptr<Mesh> mesh = std::make_shared<Mesh>();`.
2.  Print `mesh.use_count()` (Should be 1).
3.  Create another pointer `auto mesh2 = mesh;` (Copying is allowed!).
4.  Print `mesh.use_count()` (Should be 2).
5.  Reset `mesh`. Check if Mesh is destroyed (It shouldn't be, `mesh2` still holds it).

### Task 3: The Cycle Problem (Weak Pointers)
1.  Define a `Node` class.
2.  Give `Node` a member `std::shared_ptr<Node> parent`.
3.  Create Parent and Child.
4.  Set `child->parent = parent` and `parent->child = child`.
5.  Observe that destructors are NEVER called! (Memory Leak).
6.  **Fix:** Change `parent` to `std::weak_ptr<Node>`.

## Verification
Compile and run.
```bash
cd todo
mkdir build && cd build
cmake ..
cmake --build .
./main
```
Output should show proper cleanup of Mesh and Nodes.
