# Exercise 07: Chessboard Detection

## Goal
Detect corners in a standard calibration chessboard pattern.

## Learning Objectives
1.  Use `cv::findChessboardCorners`.
2.  Refine corner locations with `cv::cornerSubPix`.
3.  Visualize detections with `cv::drawChessboardCorners`.

## Analogy: The Surveyor's Ruler
To check if a camera sees correctly (Calibration), you need to show it something where you know *exactly* what it looks like.
*   **The Problem:** The world is messy. Trees don't have perfect straight lines.
*   **The Solution (Chessboard):** It's the perfect ruler.
    *   **High Contrast:** Black and white squares are easy to see.
    *   **Geometry:** We know the lines are perfectly straight and the corners are 90 degrees.
    *   **Known Size:** If we print it, we know exactly how many millimeters are between corners.
*   **Sub-pixel Refinement:** Imagine looking at the corner with a microscope. Instead of saying "The corner is at pixel (100, 100)", we look at the gradients and say "It's actually at (100.4, 99.8)". This precision is crucial for good calibration.

## Practical Motivation
Camera calibration requires precise 2D-3D correspondences. A chessboard provides known 3D structure (planar grid) and easily detectable 2D features (corners).

## Theory & Background

### Chessboard
Defined by `board_size` (inner corners, e.g., 9x6).
- **Inner Corners**: The points where 4 squares meet.
- **Sub-pixel Accuracy**: Gradient-based refinement to find the corner location with precision $< 1$ pixel.

## Implementation Tasks

### Task 1: Detect
Implement `detect_chessboard(img, board_size)` returning a vector of points.

### Task 2: Refine
Apply `cornerSubPix` if corners are found.

## Common Pitfalls
- `board_size` is (columns, rows) of *inner* corners, not squares.
- Grayscale conversion is required for `cornerSubPix`.
