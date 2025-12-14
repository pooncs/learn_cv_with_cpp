#include <iostream>
#include <cmath>
#include "rotations.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main() {
    double roll = 0.1;
    double pitch = 0.2;
    double yaw = 0.3;

    std::cout << "Input Euler (rad): " << roll << ", " << pitch << ", " << yaw << "\n\n";

    std::cout << "=== Task 1: Euler to Matrix ===\n";
    Eigen::Matrix3d R = euler_to_matrix(roll, pitch, yaw);
    std::cout << "Rotation Matrix:\n" << R << "\n\n";

    std::cout << "=== Task 2: Matrix to Axis-Angle ===\n";
    Eigen::AngleAxisd aa = matrix_to_axis_angle(R);
    std::cout << "Axis: " << aa.axis().transpose() << "\n";
    std::cout << "Angle: " << aa.angle() << " rad\n\n";

    return 0;
}
