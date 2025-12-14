#include <iostream>
#include <cassert>
#include "transformer.hpp"

void test_project() {
    Transformer t;
    // Simple intrinsics: f=100, c=50
    t.set_intrinsics(100, 100, 50, 50);
    // Identity Extrinsics
    t.set_extrinsics(Eigen::Matrix4d::Identity());

    // Point at (0, 0, 10)
    // x = f*X/Z + cx = 100*0/10 + 50 = 50
    // y = f*Y/Z + cy = 100*0/10 + 50 = 50
    Eigen::Vector3d p(0, 0, 10);
    Eigen::Vector2d pix = t.project(p);

    assert(std::abs(pix(0) - 50.0) < 1e-4);
    assert(std::abs(pix(1) - 50.0) < 1e-4);

    // Point at (1, 1, 10)
    // x = 100*1/10 + 50 = 60
    Eigen::Vector3d p2(1, 1, 10);
    Eigen::Vector2d pix2 = t.project(p2);
    assert(std::abs(pix2(0) - 60.0) < 1e-4);

    std::cout << "[PASS] project\n";
}

void test_back_project() {
    Transformer t;
    t.set_intrinsics(100, 100, 50, 50);
    t.set_extrinsics(Eigen::Matrix4d::Identity());

    // Pixel (60, 60), Depth 10
    // x = (60-50)*10/100 = 1
    Eigen::Vector2d pix(60, 60);
    Eigen::Vector3d p = t.back_project(pix, 10.0);

    assert(std::abs(p(0) - 1.0) < 1e-4);
    assert(std::abs(p(1) - 1.0) < 1e-4);
    assert(std::abs(p(2) - 10.0) < 1e-4);

    std::cout << "[PASS] back_project\n";
}

int main() {
    test_project();
    test_back_project();
    return 0;
}
