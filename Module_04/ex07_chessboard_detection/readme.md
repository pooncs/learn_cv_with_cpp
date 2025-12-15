# Exercise 07: Chessboard Detection

## Goal
Detect corners in a standard calibration chessboard pattern.

## Learning Objectives
1.  Use `cv::findChessboardCorners`.
2.  Refine corner locations with `cv::cornerSubPix`.
3.  Visualize detections with `cv::drawChessboardCorners`.

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
