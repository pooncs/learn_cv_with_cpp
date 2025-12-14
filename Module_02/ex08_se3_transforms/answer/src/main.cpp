#include <iostream>
#include "se3.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main() {
    // T1: Translate X+1
    Eigen::Matrix3d R1 = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t1(1, 0, 0);
    Eigen::Matrix4d T1 = create_se3(R1, t1);

    // T2: Rotate Z 90 deg
    Eigen::Matrix3d R2 = Eigen::AngleAxisd(M_PI/2, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    Eigen::Vector3d t2(0, 0, 0);
    Eigen::Matrix4d T2 = create_se3(R2, t2);

    std::cout << "T1 (Translate X+1):\n" << T1 << "\n\n";
    std::cout << "T2 (Rotate Z 90):\n" << T2 << "\n\n";

    // Point p at origin
    Eigen::Vector3d p(0, 0, 0);

    // Apply T1 then T2: p' = T2 * (T1 * p)
    // T1 moves to (1,0,0). T2 rotates (1,0,0) around Z -> (0,1,0)
    Eigen::Vector3d p_prime = apply_transform(T2 * T1, p);
    std::cout << "T2 * T1 * p: " << p_prime.transpose() << "\n";

    // Apply T2 then T1: p'' = T1 * (T2 * p)
    // T2 rotates (0,0,0) -> (0,0,0). T1 moves to (1,0,0).
    Eigen::Vector3d p_double_prime = apply_transform(T1 * T2, p);
    std::cout << "T1 * T2 * p: " << p_double_prime.transpose() << "\n";

    return 0;
}
