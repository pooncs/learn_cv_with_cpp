# Exercise 08: Rigid Body Transforms (SE3)

## Goal
Construct $4 \times 4$ transformation matrices (SE3) to chain robot joint poses.

## Learning Objectives
1.  **Homogeneous Coordinates:** Why we need that extra "1" at the end of vectors.
2.  **SE(3) Construction:** Building the big matrix from Rotation and Translation.
3.  **Forward Kinematics:** Chaining matrices to find where the robot hand is.

## Analogy: The Robot Arm (Lego Stacking)
*   **The Problem:** Where is the robot's hand?
*   **SE(3) Matrix:** Represents one "Lego Block" or Joint.
    *   "The Elbow is attached 10cm away from the Shoulder and rotated 30 degrees."
*   **Chaining ($T_{total} = T_1 \times T_2 \times T_3$):** Stacking the blocks.
    *   To find the Hand's position in the World, you multiply:
    *   $T_{World\_Hand} = T_{World\_Shoulder} \times T_{Shoulder\_Elbow} \times T_{Elbow\_Hand}$.

## Practical Motivation
*   **Robotics:** Calculating end-effector position.
*   **AR/VR:** Placing a virtual cup on a table (Camera-to-Object transform).
*   **SLAM:** Tracking camera movement ($Pose_{t} = Pose_{t-1} \times \Delta Pose$).

## Step-by-Step Instructions

### Task 1: Create SE3 Helper
Open `src/main.cpp`.
*   Implement `createSE3(Rotation, Translation)`.
    *   Input: `Eigen::Matrix3d R`, `Eigen::Vector3d t`.
    *   Output: `Eigen::Matrix4d T`.
    *   Structure:
        ```
        [ R  t ]
        [ 0  1 ]
        ```
    *   *Hint:* Use `.block()` or comma initializer.

### Task 2: Transform a Point
*   Implement `transformPoint(T, p)`.
    *   Input: `Eigen::Matrix4d T`, `Eigen::Vector3d p`.
    *   Output: `Eigen::Vector3d p_new`.
    *   Method:
        1.  Convert $p$ to homogeneous $p_h = [x, y, z, 1]^T$.
        2.  Multiply $p'_h = T \times p_h$.
        3.  Extract first 3 components (divide by w if w!=1, but for rigid bodies w=1).
    *   *Shortcut:* In Eigen, `T * p.homogeneous()` works if you cast back.

### Task 3: Chain Transforms (Non-Commutative!)
*   Create $T_1$: Translation (1, 0, 0).
*   Create $T_2$: Rotation Z (90 deg).
*   Compute $A = T_1 \times T_2$.
*   Compute $B = T_2 \times T_1$.
*   Apply both to point origin $(0,0,0)$.
*   *Observation:* $A$ means "Rotate then Translate" (in global frame) or "Translate then Rotate" (in local)?
    *   Standard Rule: $T_A T_B p$ applies $T_B$ first, then $T_A$.

## Verification
Compile and run.
```bash
cd todo
mkdir build && cd build
cmake ..
cmake --build .
./main
```
