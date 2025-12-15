# Exercise 02: Depth to Cloud

## Goal
Convert a depth image + intrinsics into a 3D point cloud.

## Learning Objectives
1.  Understand the relationship between Depth map and 3D points.
2.  Perform back-projection for every pixel efficiently.
3.  Visualize the result (write to PLY).

## Theory & Background

### Back-Projection
For a pixel $(u, v)$ with depth $Z$:
$$ X = (u - c_x) \cdot Z / f_x $$
$$ Y = (v - c_y) \cdot Z / f_y $$
$$ Z = Z $$

Input: `cv::Mat` (Depth, usually `CV_16U` in mm or `CV_32F` in meters).
Output: `std::vector<Point3D>`.

## Implementation Tasks

### Task 1: Depth to Points
Implement `depth_to_cloud(depth_img, K)` function.
Handle invalid depth values (0 or NaN).

## Common Pitfalls
- Unit conversion (mm vs meters).
- Coordinate system (Y-down vs Y-up).
