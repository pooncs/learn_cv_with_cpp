# Exercise 03: Block Operations

## Goal
Extract and manipulate Regions of Interest (ROI) using `.block()`, `.row()`, and `.col()`.

## Learning Objectives
1.  **Block Access:** Extract sub-matrices efficiently.
2.  **Fixed vs. Dynamic Blocks:** Use templates `.block<rows, cols>(i, j)` for speed when size is known.
3.  **In-Place Modification:** Writing to a block updates the original matrix.

## Analogy: The Window Frame
*   **The Matrix:** A large wall of numbered bricks.
*   **The Block (`.block()`):** You hold up a rectangular **Window Frame** against the wall.
    *   You can read the numbers inside the frame.
    *   If you spray paint inside the frame, **the actual wall changes**.
    *   Moving the frame (changing indices) lets you access different parts without rebuilding the wall.

## Practical Motivation
In Computer Vision:
*   **Cropping:** Extracting a face from a photo.
*   **Patching:** Copying a watermark into an image.
*   **Pose Extraction:** Taking the top-left $3 \times 3$ (Rotation) from a $4 \times 4$ Transformation matrix.

## Step-by-Step Instructions

### Task 1: Extract 2x2 Block
Open `src/main.cpp`.
*   Create a $4 \times 4$ matrix filled with numbers (e.g., 0 to 15).
*   **Goal:** Extract the center $2 \times 2$ block (Starting at row 1, col 1).
*   Use `m.block<2, 2>(1, 1)` (Fixed size) or `m.block(1, 1, 2, 2)` (Dynamic).
*   Print it.

### Task 2: Set Row to Zero
*   **Goal:** Clear the 3rd row (index 2).
*   Use `m.row(2).setZero()`.
*   Print the modified matrix.

### Task 3: Paste Block
*   Create a small $2 \times 2$ matrix of Ones.
*   **Goal:** Paste it into the bottom-right corner of the $4 \times 4$ matrix.
*   Use `m.bottomRightCorner(2, 2) = smallMat;` or `m.block(2, 2, 2, 2) = smallMat;`.
*   Print the result.

## Verification
Compile and run.
```bash
cd todo
mkdir build && cd build
cmake ..
cmake --build .
./main
```
