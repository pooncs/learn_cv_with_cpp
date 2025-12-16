# Exercise 10: RAII Wrapper (Rule of Five)

## Goal
Create a resource wrapper (RAII) for a C-style file handle. You will ensure the file is closed automatically when the wrapper goes out of scope, handling copying and moving correctly.

## Learning Objectives
1.  **RAII (Resource Acquisition Is Initialization):** Constructor acquires, Destructor releases.
2.  **Rule of Five:** If you define one of (Destructor, Copy Ctor, Copy Assign, Move Ctor, Move Assign), you likely need all five.
3.  wrapping C APIs (`FILE*`, `cudaMalloc`, `open`) safely.

## Analogy: The Magic Backpack (Library Book)
*   **Old C++ (Manual Management):**
    *   You walk to the library, get a book (`fopen`).
    *   You read it.
    *   You MUST walk back to return it (`fclose`).
    *   *Risk:* If you get hit by lightning (Exception) while reading, the book is never returned.
*   **Modern C++ (RAII):**
    *   You have a **Magic Backpack**.
    *   When you put the book in the backpack (Constructor), it's yours.
    *   When the backpack vanishes (Destructor/Scope End), the book **automatically teleports** back to the library.
    *   *Move:* You can hand the backpack to a friend. Now they are responsible.
    *   *Copy:* You generally CANNOT photocopy the book (Delete Copy Constructor), or if you do, you need a special process.

## Practical Motivation
We often use C libraries (OpenCV C-API, CUDA, OS calls) that give us "Handles" or pointers that we must manually release.
*   `FILE* f = fopen(...)` -> `fclose(f)`
*   `void* ptr; cudaMalloc(&ptr)` -> `cudaFree(ptr)`
*   `int socket = open(...)` -> `close(socket)`

Wrapping these in a class ensures we never leak resources, even if an exception occurs.

## Step-by-Step Instructions

### Task 1: The Wrapper Class
Open `src/main.cpp`. Define `class FileHandle`.
*   Member: `FILE* file = nullptr;`
*   Constructor: Takes a filename, calls `fopen`. Check if null.
*   Destructor: Calls `fclose` if `file` is not null.

### Task 2: Rule of Five - Delete Copy
It makes no sense to copy a file handle (closing it twice causes a crash).
*   **Task:** Delete Copy Constructor and Copy Assignment.
    ```cpp
    FileHandle(const FileHandle&) = delete;
    FileHandle& operator=(const FileHandle&) = delete;
    ```

### Task 3: Rule of Five - Implement Move
We *do* want to pass file handles around.
*   **Task:** Implement Move Constructor and Move Assignment.
    *   Steal the `FILE*`.
    *   Set source `FILE*` to `nullptr`.

### Task 4: Usage
In `main()`:
1.  Create a `FileHandle`.
2.  Write to it using `fprintf`.
3.  Pass it to a function `process_log(FileHandle f)` (requires move).
4.  Verify file is closed (add print in destructor).

## Verification
Compile and run.
```bash
cd todo
mkdir build && cd build
cmake ..
cmake --build .
./main
```
Output should show "File Opened", "Writing...", "File Closed".
