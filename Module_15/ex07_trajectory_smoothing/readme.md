# Exercise 07: Trajectory Smoothing with B-Splines

## Goal
Take a jagged path (e.g., from A* or RRT) and smooth it into a continuous curve suitable for robot control.

## Learning Objectives
1.  **B-Splines:** Understand Control Points, Knots, and Basis Functions.
2.  **Continuity:** $C^0$ (connected), $C^1$ (tangent continuous), $C^2$ (curvature continuous).
3.  **Optimization:** Fit a spline that stays close to the original waypoints.

## Practical Motivation
Robots cannot turn 90 degrees instantly (infinite acceleration). Smooth paths allow for higher speeds and less mechanical wear.

## Step-by-Step Instructions
1.  Input: A list of points (waypoints).
2.  Use a library (or simple implementation) to generate a Cubic B-Spline using these points as control points.
3.  Sample the spline at fine intervals to draw the smooth curve.

## Verification
*   Visual check: The curve should flow near the waypoints without sharp corners.
