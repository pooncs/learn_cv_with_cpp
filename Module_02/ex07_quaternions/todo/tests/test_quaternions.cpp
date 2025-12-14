#include <iostream>
#include <cassert>
#include <cmath>
#include "quaternions.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void test_slerp() {
    Eigen::Quaterniond q1 = Eigen::Quaterniond::Identity(); // 0 deg
    Eigen::Quaterniond q2(Eigen::AngleAxisd(M_PI/2, Eigen::Vector3d::UnitZ())); // 90 deg Z

    // t = 0.5 -> Should be 45 deg Z
    Eigen::Quaterniond q_mid = custom_slerp(q1, q2, 0.5);
    Eigen::AngleAxisd aa(q_mid);

    assert(std::abs(aa.angle() - M_PI/4) < 1e-4);
    assert(std::abs(aa.axis()(2) - 1.0) < 1e-4);

    std::cout << "[PASS] custom_slerp\n";
}

int main() {
    test_slerp();
    return 0;
}
