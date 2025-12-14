# Exercise 03: Color Spaces

## Goal
Implement manual RGB-to-Grayscale and RGB-to-HSV conversion kernels.

## Learning Objectives
1.  Understand how color images are represented (BGR in OpenCV).
2.  Implement pixel-wise transformation logic.
3.  Understand the math behind Grayscale (Luma) and HSV (Hue, Saturation, Value).

## Practical Motivation
- **Grayscale**: Reduces dimensionality for processing (edge detection, descriptors).
- **HSV**: Separates color (Hue) from intensity (Value), making color-based tracking robust to lighting changes.

## Theory & Background

### RGB to Grayscale
$$ Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B $$
Note: OpenCV uses BGR order, so `img.at<Vec3b>(y,x)[0]` is Blue.

### RGB to HSV
HSV is a cylindrical geometry.
- **Value**: $V = \max(R, G, B)$
- **Saturation**: $S = \begin{cases} 0 & \text{if } V=0 \\ \frac{V - \min(R,G,B)}{V} & \text{otherwise} \end{cases}$
- **Hue**: Angle around the axis.

## Implementation Tasks

### Task 1: to_gray
Iterate over the BGR image and produce a single-channel grayscale image using the formula.

### Task 2: to_hsv
Implement RGB to HSV conversion manually. Compare with `cv::cvtColor`.

## Common Pitfalls
- BGR vs RGB.
- Integer division (use floats for calculation, then cast back).
- Hue range: OpenCV maps $0..360$ to $0..180$ to fit in `uint8`.
