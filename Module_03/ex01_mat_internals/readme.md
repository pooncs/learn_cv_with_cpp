# Exercise 01: cv::Mat Internals

## Goal
Deep dive into `cv::Mat` memory layout, reference counting, and basic properties.

## Learning Objectives
1.  Understand how `cv::Mat` stores data (Header vs Data Pointer).
2.  Inspect `rows`, `cols`, `channels`, `step`, and `elemSize`.
3.  Understand Reference Counting (shallow copy vs deep copy).

## Practical Motivation
`cv::Mat` is the fundamental data structure in OpenCV. Misunderstanding how it manages memory leads to:
- **Memory Leaks**: If not handled correctly (though RAII helps).
- **Unintended Modifications**: Modifying a shallow copy changes the original image.
- **Segfaults**: Accessing data with wrong step/stride.

## Theory & Background

### Memory Layout
- **Header**: Contains metadata (size, type, pointer to data). Small footprint.
- **Data**: The actual pixel array. Large footprint.
- **Ref Count**: When you copy a Mat (`B = A`), the data is NOT copied. `B` points to the same data, and ref count increases. Use `A.clone()` for deep copy.

### Stride (Step)
- `step[0]`: Bytes per row (width * channels * bytes_per_channel + padding).
- `step[1]`: Bytes per element (channels * bytes_per_channel).

## Implementation Tasks

### Task 1: Create Mat Manually
Allocate a 3x3 RGB image (3 channels) manually.

### Task 2: Inspect Properties
Print `rows`, `cols`, `channels`, `elemSize` (bytes per pixel), and `step` (bytes per row).

### Task 3: Reference Counting
Create `A`. Assign `B = A`. Modify `B`. Check if `A` changed.
Create `C = A.clone()`. Modify `C`. Check if `A` changed.

## Common Pitfalls
- Assuming `step` equals `width * channels`. Sometimes OpenCV adds padding for alignment.
- Confusing `type()` (CV_8UC3) with `depth()` (CV_8U).
