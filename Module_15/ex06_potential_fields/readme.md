# Exercise 06: Artificial Potential Fields

## Goal
Implement a reactive local planner using Potential Fields.

## Learning Objectives
1.  **Attraction:** Goal exerts an attractive force.
2.  **Repulsion:** Obstacles exert repulsive forces.
3.  **Gradient Descent:** The robot moves along the negative gradient of the total potential.
4.  **Local Minima:** Understand the main drawback (getting stuck).

## Practical Motivation
Potential fields are computationally cheap and great for real-time collision avoidance, though they can get stuck in U-shaped obstacles.

## Step-by-Step Instructions
1.  Define $U_{att}(q) = \frac{1}{2} k_{att} d(q, q_{goal})^2$.
2.  Define $U_{rep}(q) = \frac{1}{2} k_{rep} (\frac{1}{d(q, obs)} - \frac{1}{d_0})^2$ if close to obstacle.
3.  Compute Force $F = -\nabla (U_{att} + U_{rep})$.
4.  Update position: $q_{t+1} = q_t + \alpha F$.

## Verification
*   Robot should smoothly curve around a single obstacle to reach the goal.
*   Place an obstacle directly between start and goal to see if it avoids it.
