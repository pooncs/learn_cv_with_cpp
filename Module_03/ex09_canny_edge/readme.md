# Exercise 09: Canny Edge Detection (Non-Maximum Suppression)

## Goal
Step-by-step implementation of Non-Maximum Suppression (NMS) for Canny Edge Detection.

## Learning Objectives
1.  Understand the Canny Edge Detection pipeline.
2.  Implement NMS to thin edges to 1-pixel width.
3.  Understand Hysteresis Thresholding (conceptually).

## Practical Motivation
Raw gradients produce thick edges. NMS thins them by keeping only the local maximum along the gradient direction. This is critical for precise edge localization.

## Theory & Background

### The Canny Pipeline
1.  **Gaussian Blur**: Reduce noise.
2.  **Gradients**: Compute magnitude $M$ and orientation $\theta$.
3.  **Non-Maximum Suppression**:
    For each pixel $(x,y)$, check neighbors in the direction of the gradient $\theta$.
    If $M(x,y)$ is smaller than neighbors, set to 0.
4.  **Hysteresis Thresholding**: Use high and low thresholds to link edges.

### NMS Logic
Quantize $\theta$ to 4 directions:
- 0 deg (Horizontal) -> Check Left/Right neighbors.
- 45 deg (Diagonal) -> Check TopRight/BottomLeft.
- 90 deg (Vertical) -> Check Top/Bottom.
- 135 deg (Diagonal) -> Check TopLeft/BottomRight.

## Implementation Tasks

### Task 1: NMS
Implement a function that takes Magnitude and Angle maps, and returns the thinned edge map.

## Common Pitfalls
- **Interpolation**: Ideally, NMS interpolates neighbor values for exact gradient direction. Simplifying to 8-neighborhood (0, 45, 90, 135) is a common approximation.
- **Border Handling**: Skip borders to avoid out-of-bounds access.
