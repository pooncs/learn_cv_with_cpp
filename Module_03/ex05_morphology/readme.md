# Exercise 05: Morphology

## Goal
Implement Dilation and Erosion manually on binary images.

## Learning Objectives
1.  **Set Operations:** Treating images as sets of pixels.
2.  **Structuring Element:** The shape of the tool (Square, Circle, Cross).
3.  **Opening & Closing:** Combining operations to clean noise.

## Analogy: The Virus and the Siege
*   **Dilation (The Virus):**
    *   **Rule:** If *any* of your neighbors are sick (White/255), you get sick too.
    *   **Effect:** White regions grow. Small black holes are filled.
*   **Erosion (The Siege):**
    *   **Rule:** You can only survive if *all* your neighbors are allies (White/255). If even one enemy (Black/0) touches you, you fall.
    *   **Effect:** White regions shrink. Small white noise dots disappear.

## Practical Motivation
*   **Noise Removal:** "Opening" (Erosion followed by Dilation) removes salt noise (white dots) without changing the size of large objects.
*   **Hole Filling:** "Closing" (Dilation followed by Erosion) fills pepper noise (black holes).
*   **Boundary Extraction:** `Dilated - Original` gives you the outline of the object.

## Step-by-Step Instructions

### Task 1: Dilate
Open `src/main.cpp`.
*   Implement `custom_dilate(Mat& src, Mat& dst)`.
*   Structuring Element: $3 \times 3$ square.
*   Logic:
    *   For each pixel $(y, x)$:
    *   Check 8 neighbors + center.
    *   If `max(neighbors) == 255`, set `dst(y, x) = 255`.
    *   Else `0`.

### Task 2: Erode
*   Implement `custom_erode(Mat& src, Mat& dst)`.
*   Logic:
    *   For each pixel $(y, x)$:
    *   Check 8 neighbors + center.
    *   If `min(neighbors) == 255` (All are white), set `dst(y, x) = 255`.
    *   Else `0`.

### Task 3: Opening (Noise Removal)
*   Create an image with random noise.
*   Apply `custom_erode` then `custom_dilate`.
*   Verify noise is gone.

## Verification
Compile and run.
```bash
cd todo
mkdir build && cd build
cmake ..
cmake --build .
./main
```
