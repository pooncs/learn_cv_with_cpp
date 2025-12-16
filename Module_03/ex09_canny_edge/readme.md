# Exercise 09: Canny Edge Detection (Non-Maximum Suppression)

## Goal
Step-by-step implementation of Non-Maximum Suppression (NMS) for Canny Edge Detection.

## Learning Objectives
1.  **The Pipeline:** Blur -> Gradient -> NMS -> Hysteresis.
2.  **Thinning:** Why gradients give "thick" edges and how to make them 1-pixel wide.
3.  **Directional Check:** Checking neighbors based on gradient angle.

## Analogy: The Ridge Walker
*   **Gradient Magnitude:** Imagine the image is a mountain range. Edges are the sharp ridges.
*   **The Problem:** The ridge is 5 meters wide (Thick edge). We want a thin line on a map.
*   **NMS (The Test):** You stand at a point on the ridge.
    *   Look perpendicular to the ridge (Gradient Direction).
    *   If you step forward or backward, do you go downhill?
    *   **Yes:** You are at the **Local Peak**. You are the Edge.
    *   **No (One neighbor is higher):** You are just on the slope. You are NOT the edge. **Suppress (Set to 0).**

## Practical Motivation
*   **Precision:** Thick edges are bad for measurements (e.g., "How wide is this part?").
*   **Feature Matching:** We want precise locations for corners and lines.

## Step-by-Step Instructions

### Task 1: Setup
Open `src/main.cpp`.
*   Load image, convert to Gray, apply GaussianBlur.
*   Compute Gradients ($G_x, G_y$), Magnitude ($M$), and Angle ($\theta$).
*   *Hint:* Use `cv::cartToPolar(Gx, Gy, mag, angle, true)`.

### Task 2: Quantize Angles
*   NMS usually simplifies directions to 4 cases (0, 45, 90, 135 degrees).
*   Map $\theta$ to these 4 bins.
    *   $(-22.5, 22.5) \to 0^\circ$ (Horizontal gradient -> Vertical Edge).
    *   $(22.5, 67.5) \to 45^\circ$.
    *   etc.

### Task 3: Implement NMS
*   Create `nms_img` initialized to zeros.
*   Loop over pixels (skip borders).
*   For each pixel $(y, x)$:
    *   Get direction $d$.
    *   Get values of two neighbors along $d$ (e.g., if $0^\circ$, neighbors are $(y, x-1)$ and $(y, x+1)$).
    *   If $M(y,x) \ge M(neighbor1)$ AND $M(y,x) \ge M(neighbor2)$:
        *   `nms_img(y,x) = M(y,x)` (Keep it).
    *   Else:
        *   `nms_img(y,x) = 0` (Suppress it).

### Task 4: Compare
*   Run `cv::Canny` on the original image.
*   Compare your NMS result (which is not binary yet, it still has float magnitudes) with Canny output.

## Verification
Compile and run.
```bash
cd todo
mkdir build && cd build
cmake ..
cmake --build .
./main
```
The NMS output should look like thin "skeleton" lines of the edges.
