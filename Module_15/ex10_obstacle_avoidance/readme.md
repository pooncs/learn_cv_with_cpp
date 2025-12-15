# Exercise 10: Dynamic Obstacle Avoidance

## Goal
Simulate a simple local planner (like Dynamic Window Approach - DWA lite) that avoids moving obstacles.

## Learning Objectives
1.  **Velocity Space:** Planning in $(v, \omega)$ space rather than $(x, y)$.
2.  **Predictive Modeling:** Projecting robot trajectory forward for short time $T$.
3.  **Scoring:** Evaluating trajectories based on clearance, heading, and velocity.

## Practical Motivation
Global planners (A*) provide a path, but local planners handle the execution and react to people walking in front of the robot.

## Step-by-Step Instructions
1.  **Motion Model:** Simple unicycle model $x' = v \cos \theta, y' = v \sin \theta, \theta' = \omega$.
2.  **Sampling:** Generate pairs of $(v, \omega)$.
3.  **Simulation:** For each pair, simulate 2 seconds forward.
4.  **Check:** Discard if hits obstacle.
5.  **Score:** Pick best trajectory (closest to goal heading, fastest speed).

## Verification
*   Robot should navigate a corridor with a moving "person" (circle).
