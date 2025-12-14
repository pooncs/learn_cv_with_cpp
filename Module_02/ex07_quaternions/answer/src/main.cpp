#include <iostream>
#include "quaternions.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main() {
    // Rotation 1: Identity (0 deg)
    Eigen::Quaterniond q1 = Eigen::Quaterniond::Identity();

    // Rotation 2: 90 deg around Z
    Eigen::Quaterniond q2(Eigen::AngleAxisd(M_PI/2, Eigen::Vector3d::UnitZ()));

    std::cout << "q1 (0 deg): " << q1.coeffs().transpose() << "\n";
    std::cout << "q2 (90 deg Z): " << q2.coeffs().transpose() << "\n\n";

    std::cout << "=== Task: SLERP interpolation ===\n";
    for(int i=0; i<=4; ++i) {
        double t = i / 4.0; // 0, 0.25, 0.5, 0.75, 1.0
        Eigen::Quaterniond qt = custom_slerp(q1, q2, t);
        
        // Convert to axis angle to verify angle
        Eigen::AngleAxisd aa(qt);
        std::cout << "t=" << t << " -> Angle: " << aa.angle() * 180.0 / M_PI << " deg\n";
    }

    return 0;
}
