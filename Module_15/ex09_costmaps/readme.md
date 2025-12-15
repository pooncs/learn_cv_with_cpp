# Exercise 09: Costmaps & Inflation

## Goal
Generate a costmap from a binary occupancy grid, adding a "risk gradient" around obstacles.

## Learning Objectives
1.  **Inflation Radius:** Distance within which robot is definitely in collision.
2.  **Cost Decay:** Exponential or linear decay of cost as distance from obstacle increases.
3.  **Lethal vs Non-Lethal:** Distinguishing between "wall" (254) and "near wall" (1-253).

## Practical Motivation
Instead of binary (Free/Occupied), robots prefer to stay away from walls. A costmap tells the planner "You *can* go here, but it's expensive/risky."

## Step-by-Step Instructions
1.  Input: Binary grid (0 or 1).
2.  Compute Distance Transform (Euclidean distance to nearest obstacle for every pixel).
3.  Map distance to cost:
    *   If dist < robot_radius: Cost = LETHAL (254).
    *   Else: Cost = $k \cdot \exp(-\alpha \cdot dist)$.

## Verification
*   Visualize the costmap as a grayscale image. It should look like blurred obstacles.
