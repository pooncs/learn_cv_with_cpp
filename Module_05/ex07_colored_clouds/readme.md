# Exercise 07: Colored Clouds

## Goal
Map RGB texture onto a generated point cloud.

## Learning Objectives
1.  Understand (u, v) mapping for each 3D point.
2.  Sample color from an image.
3.  Store (x, y, z, r, g, b).

## Theory & Background

### Texture Mapping
For a point $P = (X, Y, Z)$ in camera coordinates:
$$ u = f_x \frac{X}{Z} + c_x $$
$$ v = f_y \frac{Y}{Z} + c_y $$

If $(u, v)$ is within image bounds, sample color $I(u, v)$ and assign to $P$.

## Implementation Tasks

### Task 1: Colorize
Implement `colorize_cloud(cloud, img, K)` returning `std::vector<PointRGB>`.

## Common Pitfalls
- Out of bounds access.
- BGR vs RGB.
