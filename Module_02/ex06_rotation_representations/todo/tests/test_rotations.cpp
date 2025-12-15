#include <iostream>
#include <cassert>
#include <cmath>
#include "rotations.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void test_euler_to_matrix() {
    // 90 deg rotation around Z
    double roll = 0;
    double pitch = 0;
    double yaw = M_PI / 2.0;

    Eigen::Matrix3d R = euler_to_matrix(roll, pitch, yaw);
    
    // Rz(90) = [0 -1 0; 1 0 0; 0 0 1]
    assert(std::abs(R(0,0)) < 1e-6);
    assert(std::abs(R(0,1) + 1.0) < 1e-6);
    assert(std::abs(R(1,0) - 1.0) < 1e-6);
    assert(std::abs(R(2,2) - 1.0) < 1e-6);

    std::cout << "[PASS] euler_to_matrix\n";
}

void test_matrix_to_axis_angle() {
    Eigen::Matrix3d R = Eigen::AngleAxisd(M_PI/2.0, Eigen::Vector3d::UnitY()).toRotationMatrix();
    Eigen::AngleAxisd aa = matrix_to_axis_angle(R);

    assert(std::abs(aa.angle() - M_PI/2.0) < 1e-6);
    assert(std::abs(aa.axis()(1) - 1.0) < 1e-6); // Y axis

    std::cout << "[PASS] matrix_to_axis_angle\n";
}

int main() {
    test_euler_to_matrix();
    test_matrix_to_axis_angle();
    return 0;
}
