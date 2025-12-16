# Exercise 09: Stereo Rectification

## Goal
Compute rectification transforms given two camera matrices.

## Learning Objectives
1.  Understand Epipolar Geometry basics.
2.  Use `cv::stereoRectify` to align epipolar lines horizontally.
3.  Compute Undistort+Rectify maps.

## Analogy: The Binoculars Alignment
*   **Stereo Vision:** You have two eyes (cameras).
*   **The Problem:** If one eye looks slightly up and the other looks slightly down (squinting), it's incredibly hard to match images. "Is that tree in my left eye the same as the one in my right eye?" You have to search everywhere.
*   **Rectification:** It's like mechanically locking both eyes to look perfectly straight ahead, parallel to each other.
*   **The Benefit:** Now, to find the match for a pixel in the left eye, you just have to scan the **same horizontal row** in the right eye. You never have to look up or down. This makes depth calculation 100x faster.

## Practical Motivation
Stereo matching algorithms (Block Matching) assume rectified images where corresponding points lie on the same scanline ($y_L = y_R$).

## Theory & Background

### Rectification
Transforms both image planes so they are coplanar and row-aligned.
Result: Disparity is only in $x$ direction.

## Implementation Tasks

### Task 1: Compute Rectification
Given $K_1, D_1, K_2, D_2, R, T$, compute $R_1, R_2, P_1, P_2, Q$.

### Task 2: Init Maps
Use `cv::initUndistortRectifyMap` for both cameras.

## Common Pitfalls
- Image size must be consistent.
- `alpha` parameter in `stereoRectify`: 0 (zoom in, valid pixels only) vs 1 (keep all pixels, black borders).
