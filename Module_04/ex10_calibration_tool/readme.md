# Exercise 10: Calibration Tool

## Goal
Develop a complete application that calibrates a camera from a video stream.

## Learning Objectives
1.  Capture video/images.
2.  Detect calibration patterns in a loop.
3.  Collect valid frames and run `calibrateCamera`.
4.  Save results to YAML/JSON.

## Practical Motivation
This is the "Hello World" of real-world CV systems. Before doing anything with a camera, you must calibrate it.

## Implementation Tasks

### Task 1: Capture Loop
- Open webcam.
- Detect chessboard.
- If user presses 'c', save corners.

### Task 2: Calibrate
- When enough frames (e.g., 10) are collected, run calibration.
- Report RMS error.

### Task 3: Save
- Save $K$ and $D$ to `calibration.yml`.

## Common Pitfalls
- Motion blur: Don't capture frames if the board is moving fast.
- Coverage: Ensure corners cover the entire image frame (edges and corners) for good distortion estimation.
