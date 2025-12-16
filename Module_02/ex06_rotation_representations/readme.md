# Exercise 06: Rotation Representations

## Goal
Convert between Rotation Matrices (SO3), Euler Angles, and Axis-Angle representations.

## Learning Objectives
1.  **Three Languages of Rotation:** Matrix (Math), Euler (Human), Axis-Angle (Physics).
2.  **Eigen Geometry:** Using `Eigen::AngleAxis` and `Eigen::Matrix3d`.
3.  **Conversions:** Moving seamlessly between these forms.

## Analogy: The Language of Spinning
*   **Rotation Matrix (3x3):** The "Mathematical Esperanto".
    *   Uses 9 numbers to describe 3 degrees of freedom.
    *   Verbose, but **Unambiguous** and Singular-free. Math loves it.
*   **Euler Angles (Roll, Pitch, Yaw):** "Giving Directions".
    *   "Turn left, then look up, then tilt your head."
    *   Intuitive for humans.
    *   **Fatal Flaw:** If you look straight up (Pitch 90), "turning left" and "tilting head" become the same motion. This is **Gimbal Lock**.
*   **Axis-Angle:** "Spin the Globe".
    *   Pick a stick (Axis). Poke it through the center of the object. Spin the object around that stick by X degrees (Angle).
    *   Very physical interpretation.

## Practical Motivation
*   **Sensors:** IMUs often give Euler angles.
*   **Robotics:** Joints are controlled by angles.
*   **Vision:** Interpolation (smoothly rotating from A to B) is hard with Matrices and Euler angles. (See next exercise on Quaternions).

## Step-by-Step Instructions

### Task 1: Euler to Matrix
Open `src/main.cpp`.
*   Define `roll`, `pitch`, `yaw` (e.g., 30, 45, 60 degrees - convert to radians!).
*   Create a Rotation Matrix using the **ZYX convention**: $R = R_z(\text{yaw}) \cdot R_y(\text{pitch}) \cdot R_x(\text{roll})$.
    *   Use `Eigen::AngleAxisd(angle, axis).toRotationMatrix()`.
    *   Multiply them in the correct order (Left multiplication = global, Right = local. Usually we want $R_{total} = R_z R_y R_x$).

### Task 2: Matrix to Axis-Angle
*   Convert the matrix $R$ back to Axis-Angle.
*   Use `Eigen::AngleAxisd(R)`.
*   Print the `.axis()` and `.angle()`.

### Task 3: Matrix to Euler (The Danger Zone)
*   Recover the angles using `R.eulerAngles(2, 1, 0)` (Indices for Z, Y, X).
*   **Warning:** The returned angles might not match your inputs exactly (e.g., 190 deg vs -170 deg), but they represent the same rotation.

## Verification
Compile and run.
```bash
cd todo
mkdir build && cd build
cmake ..
cmake --build .
./main
```
