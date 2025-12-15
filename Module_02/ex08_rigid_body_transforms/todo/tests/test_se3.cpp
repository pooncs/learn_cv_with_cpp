#include <iostream>
#include <cassert>
#include "se3.hpp"

void test_create() {
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t(1, 2, 3);
    Eigen::Matrix4d T = create_se3(R, t);
    
    assert(T(0,3) == 1);
    assert(T(1,3) == 2);
    assert(T(2,3) == 3);
    assert(T(3,3) == 1);
    assert(T(0,0) == 1);
    std::cout << "[PASS] create_se3\n";
}

void test_apply() {
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t(1, 0, 0);
    Eigen::Matrix4d T = create_se3(R, t);
    
    Eigen::Vector3d p(0, 0, 0);
    Eigen::Vector3d p_prime = apply_transform(T, p);
    
    assert(p_prime(0) == 1);
    assert(p_prime(1) == 0);
    assert(p_prime(2) == 0);
    std::cout << "[PASS] apply_transform\n";
}

int main() {
    test_create();
    test_apply();
    return 0;
}
