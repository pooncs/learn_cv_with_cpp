# Exercise 08: Post-processing

## Goal
Parse raw output tensors from a neural network (e.g., YOLO) into meaningful bounding boxes and class labels.

## Learning Objectives
1.  **Output Structure:** Understand typical detection output formats (e.g., `[cx, cy, w, h, confidence, class_scores...]`).
2.  **Pointer Arithmetic:** Navigate flat float arrays representing 3D tensors.
3.  **Thresholding:** Filter detections by confidence score.
4.  **Conversion:** Convert center coordinates (`cx, cy, w, h`) to corner coordinates (`x1, y1, x2, y2`) for display.

## Practical Motivation
The inference engine (Ex06) just gives you a big array of floats (e.g., `1x84x8400` for YOLOv8). It doesn't tell you "There's a car here". You need to interpret these numbers to draw boxes on the screen.

**Analogy:**
*   **Inference Output:** A raw spreadsheet with thousands of rows of numbers.
*   **Post-processing:** An accountant going through the spreadsheet, highlighting the profitable deals (high confidence), converting currencies (coordinates), and summarizing the results into a report (List of Detections).

## Theory: YOLO Output
YOLOv8 typically outputs a tensor of shape `[Batch, Channels, Anchors]`.
*   **Channels (84):** `cx, cy, w, h` (4) + `class_scores` (80).
*   **Anchors (8400):** The number of potential detection candidates grid-wide.

You need to:
1.  Transpose if necessary (to iterate by anchor).
2.  Find the max class score.
3.  If `max_score > threshold`, save the box.

## Step-by-Step Instructions

### Task 1: Define Structures
1.  Define a `Detection` struct: `cv::Rect box`, `float conf`, `int classId`.

### Task 2: Parse Tensor
1.  Iterate through the output array.
2.  For each candidate (anchor):
    *   Extract class scores.
    *   Find best class and score.
    *   Filter by confidence threshold.
    *   Extract box coordinates (`cx, cy, w, h`).
    *   Convert to `x, y, w, h` (top-left).
    *   Scale coordinates back to original image size if resizing was done in Pre-processing (Ex07).

### Task 3: Store Results
1.  Push valid detections to a `std::vector<Detection>`.

## Code Hints
```cpp
// Assuming output is [1, 84, 8400]
int dimensions = 84;
int rows = 8400;

float* data = outputBuffer; // Raw pointer

for (int i = 0; i < rows; ++i) {
    // Note: Memory layout depends on the model export (transpose or not).
    // Let's assume [84, 8400] where we iterate columns.
    // Or more commonly, models are exported as [8400, 84] for easier C++ parsing.
    
    // If [Rows, Dimensions]:
    float* sample = data + i * dimensions;
    float confidence = sample[4]; // Objectness (if present) or max class score
    
    if (confidence > 0.5) {
        // ... extract box ...
    }
}
```

## Verification
Create a dummy output buffer with one "perfect" detection hidden inside. Your parser should find it and print the correct Box and Class ID.
