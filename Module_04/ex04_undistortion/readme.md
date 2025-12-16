# Exercise 04: Undistortion

## Goal
Implement image undistortion using a lookup table (remap).

## Learning Objectives
1.  Understand the "inverse mapping" approach for warping.
2.  Create `map_x` and `map_y` for `cv::remap`.
3.  Correct radial and tangential distortion in images.

## Analogy: The Ironing Board
*   **Distorted Image:** A crumpled, wrinkled shirt (or a photo taken with a fisheye lens).
*   **Undistortion:** Ironing it flat to see the true pattern.
*   **Forward Mapping (The Bad Way):** You try to push every pixel from the crumpled shirt onto a flat board.
    *   *Problem:* Some pixels overlap, and some spots get no pixels at all (holes).
*   **Inverse Mapping (The Good Way):** You start with a pristine, flat white board (The Destination).
    *   For every dot on the flat board, you ask: "Where would this dot be on the crumpled shirt?"
    *   You calculate the coordinate on the crumpled shirt (Distort it).
    *   You go to that coordinate, pick up the color, and paint it on your flat board.
    *   *Result:* A perfectly filled, smooth image with no holes.

## Theory & Background

### Inverse Mapping
To avoid holes in the destination image, we iterate over every pixel $(u_{dst}, v_{dst})$ in the output image and find the corresponding source pixel $(u_{src}, v_{src})$.

1.  **Unproject**: Convert $(u_{dst}, v_{dst})$ to normalized coordinates $(x, y)$.
2.  **Distort**: Apply the distortion model to get $(x_{dist}, y_{dist})$.
3.  **Project**: Convert $(x_{dist}, y_{dist})$ to pixel coordinates $(u_{src}, v_{src})$.
4.  **Remap**: Set $I_{dst}(u_{dst}, v_{dst}) = I_{src}(u_{src}, v_{src})$.

## Implementation Tasks

### Task 1: Build Maps
Implement `compute_undistortion_maps(K, dist_coeffs, size)` returning `map_x` and `map_y`.

### Task 2: Remap
Use `cv::remap` to apply the correction.

## Common Pitfalls
- Confusing source and destination coordinates.
- Using integer maps vs float maps (float maps allow bilinear interpolation).
