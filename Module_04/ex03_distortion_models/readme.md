# Exercise 03: Distortion Models

## Goal
Implement Radial and Tangential distortion functions.

## Learning Objectives
1.  Understand the Brown-Conrady distortion model.
2.  Implement forward distortion (Undistorted -> Distorted).
3.  Visualize the "pincushion" or "barrel" effect.

## Analogy: The Funhouse Mirror
A perfect camera is like a flat, perfect window. But real lenses are curved glass.
*   **Ideal World:** Straight lines in the world appear as straight lines on the image.
*   **Radial Distortion (The Curve):** The glass is curved like a magnifying glass or a fisheye lens.
    *   *Barrel:* The center looks big, the edges look squashed. Straight lines curve outwards.
    *   *Pincushion:* The center looks small, the corners are pulled out. Straight lines curve inwards.
*   **Tangential Distortion (The Tilt):** The lens wasn't glued perfectly flat. It's slightly tilted. Things look stretched on one side.

## Theory & Background

### Normalized Coordinates
Let $x, y$ be the normalized coordinates ($x = X_c/Z_c, y = Y_c/Z_c$).
Let $r^2 = x^2 + y^2$.

### Radial Distortion
Depends on the distance from the center.
$$ x_{rad} = x (1 + k_1 r^2 + k_2 r^4 + k_3 r^6) $$
$$ y_{rad} = y (1 + k_1 r^2 + k_2 r^4 + k_3 r^6) $$

### Tangential Distortion
Due to lens not being perfectly parallel to the image plane.
$$ x_{tan} = 2p_1 x y + p_2(r^2 + 2x^2) $$
$$ y_{tan} = p_1(r^2 + 2y^2) + 2p_2 x y $$

### Total Distortion
$$ x_{dist} = x_{rad} + x_{tan} $$
$$ y_{dist} = y_{rad} + y_{tan} $$

Final pixel coordinates:
$$ u = f_x \cdot x_{dist} + c_x $$
$$ v = f_y \cdot y_{dist} + c_y $$

## Implementation Tasks

### Task 1: Distort Point
Implement `distort_point(x, y, k1, k2, p1, p2)` returning the distorted normalized point.

### Task 2: Project with Distortion
Combine pinhole projection with distortion.

## Common Pitfalls
- Applying distortion to pixel coordinates $(u,v)$ instead of normalized coordinates $(x,y)$.
