# Exercise 04: Convolution

## Goal
Implement a 3x3 convolution from scratch and verify against `cv::filter2D`.

## Learning Objectives
1.  Understand the sliding window mechanism of convolution.
2.  Handle image boundaries (Padding).
3.  Understand Correlation vs Convolution (Kernel flipping).

## Practical Motivation
Convolution is the backbone of Image Processing (Blurring, Sharpening, Edge Detection) and CNNs.

## Theory & Background

### Convolution
For a pixel $(x, y)$, the output is the weighted sum of its neighbors.
$$ G(x, y) = \sum_{dx=-1}^{1} \sum_{dy=-1}^{1} I(x+dx, y+dy) \cdot K(dx, dy) $$

### Padding
To keep the output size same as input, we pad the borders (usually with zeros or by replicating edge pixels).

## Implementation Tasks

### Task 1: convolve
Implement a function that takes a single-channel image and a $3 \times 3$ kernel, and returns the convolved image (same size). Use Zero Padding.

### Task 2: Verify
Compare your result with `cv::filter2D`.

## Common Pitfalls
- Index out of bounds at borders.
- Saturation: The sum might exceed 255. Use float for intermediate calculation and then clamp/cast.
- `cv::filter2D` actually performs **Correlation** (no kernel flip). We will implement Correlation as well (standard in CV terminology).
