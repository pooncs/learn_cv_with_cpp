# Exercise 08: Post-processing

## Goal
Parse raw output tensors from the neural network into meaningful detection results (Bounding Boxes, Classes, Scores).

## Learning Objectives
1.  Understand YOLO output format (e.g., `[Batch, 4+Classes, Boxes]`).
2.  Transpose/Reshape tensors if necessary.
3.  Filter by confidence threshold.

## Practical Motivation
The model outputs a massive array of floats. We need to convert indices back to coordinates relative to the original image.

## Theory: YOLO Output
Typical YOLOv8 output is `1 x 84 x 8400`.
*   8400: Number of candidate boxes (anchors).
*   84: 4 coordinates (cx, cy, w, h) + 80 class probabilities.
*   Note: Sometimes it's transposed. We assume `[1, 84, 8400]`.

## Step-by-Step Instructions

### Task 1: Define Struct
Open `todo/src/main.cpp`.
1.  Define `struct Detection { cv::Rect box; float conf; int classId; };`.

### Task 2: Parse Tensor
1.  Create a dummy output vector simulating `1x6x10` (1 batch, 4 coords + 2 classes, 10 boxes).
2.  Iterate through the 10 boxes.
3.  For each box:
    *   Get `cx, cy, w, h`.
    *   Find max score among classes.
    *   If score > threshold, save it.

### Task 3: Convert Coordinates
1.  `cx, cy` are usually relative to the model input size (e.g. 640).
2.  Convert to top-left `x, y`.
3.  Store in `Detection` struct.

## Verification
Print the list of detected objects.
