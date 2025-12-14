# Exercise 07: Thresholding

## Goal
Implement global and adaptive (local mean) thresholding.

## Learning Objectives
1.  Separate foreground from background using intensity.
2.  Understand why global thresholding fails on unevenly lit images.
3.  Implement adaptive thresholding using integral images or sliding windows.

## Practical Motivation
- **Document Scanning**: Binarizing text from paper.
- **Defect Detection**: Finding bright/dark spots.

## Theory & Background

### Global Thresholding
$$ dst(x,y) = \begin{cases} 255 & \text{if } src(x,y) > T \\ 0 & \text{otherwise} \end{cases} $$

### Adaptive Thresholding
The threshold $T$ varies for each pixel $(x,y)$.
$$ T(x,y) = \text{mean}(neighborhood(x,y)) - C $$
where $C$ is a constant.

## Implementation Tasks

### Task 1: Global Threshold
Implement simple binary thresholding given $T$.

### Task 2: Adaptive Threshold
Implement local mean thresholding using a window size $block\_size$ and constant $C$.
Hint: Using `cv::boxFilter` or `cv::blur` can help calculate local mean efficiently.

## Common Pitfalls
- Choosing a window size too small (noise) or too large (global-like).
- Handling borders.
