# Exercise 07: Pre-processing

## Goal
Implement image pre-processing on CPU using OpenCV to prepare data for the neural network.

## Learning Objectives
1.  Resize image to model input size (e.g., 640x640).
2.  Normalize pixel values (0-255 -> 0.0-1.0).
3.  Convert HWC (Height-Width-Channel) to NCHW (Batch-Channel-Height-Width).

## Practical Motivation
OpenCV images are `HWC` and `BGR` (uint8). Models usually expect `NCHW` and `RGB` (float). Passing raw OpenCV mats will produce garbage results.

## Theory: Layouts
*   **HWC:** `[B, G, R, B, G, R, ...]` (interleaved).
*   **CHW:** `[B, B, ...], [G, G, ...], [R, R, ...]` (planar).

## Step-by-Step Instructions

### Task 1: Load Image
Open `todo/src/main.cpp`.
1.  Load `data/lenna.png`.

### Task 2: Resize
1.  `cv::resize` to target size (e.g., 224x224).

### Task 3: Convert Color
1.  `cv::cvtColor` BGR -> RGB.

### Task 4: Normalize & Reorder
1.  Convert to float (`img.convertTo(CV_32F)`).
2.  Divide by 255.0.
3.  Use `cv::dnn::blobFromImage` OR write a manual loop to extract channels into a flat vector.
    *   Manual loop: Iterate `c` from 0-2, then `h`, then `w`.
    *   Index in vector: `c * H * W + h * W + w`.

## Verification
Print the first few values of the float vector. They should be between 0.0 and 1.0.
