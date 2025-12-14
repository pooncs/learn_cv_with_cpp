# Exercise 06: Histograms

## Goal
Compute and visualize image histograms; implement histogram equalization.

## Learning Objectives
1.  Compute the frequency distribution of pixel intensities.
2.  Normalize histograms for visualization.
3.  Implement Histogram Equalization to enhance contrast.

## Practical Motivation
Histograms tell us about exposure (under/over-exposed) and contrast. Equalization is a quick way to improve visibility in low-light images.

## Theory & Background

### Histogram
For a grayscale image (0-255), $H(i)$ is the count of pixels with intensity $i$.

### Equalization
We want to transform intensity $r$ to $s$ such that the output histogram is flat (uniform).
$$ s_k = T(r_k) = (L-1) \sum_{j=0}^{k} p_r(r_j) $$
where $L=256$, and $p_r$ is the normalized histogram (PDF). This is essentially the Cumulative Distribution Function (CDF).

## Implementation Tasks

### Task 1: Compute Histogram
Manually compute the histogram array (size 256).

### Task 2: Visualize
Draw the histogram on a blank image.

### Task 3: Equalize
Implement the CDF transformation manually and apply it.

## Common Pitfalls
- Not normalizing the CDF to range [0, 255].
- Integer overflow when summing counts (use int or float).
