#include "transformer.hpp"

Transformer::Transformer() {
    K = Eigen::Matrix3d::Identity();
    T_cw = Eigen::Matrix4d::Identity();
    T_wc = Eigen::Matrix4d::Identity();
}

void Transformer::set_intrinsics(double fx, double fy, double cx, double cy) {
    // TODO: Fill K matrix
}

void Transformer::set_extrinsics(const Eigen::Matrix4d& T_cw) {
    // TODO: Store T_cw and compute T_wc
}

Eigen::Vector2d Transformer::project(const Eigen::Vector3d& p_world) const {
    // TODO:
    // 1. Transform p_world to p_cam
    // 2. Project to p_pixel using K
    // 3. Normalize by Z
    return Eigen::Vector2d::Zero();
}

Eigen::Vector3d Transformer::back_project(const Eigen::Vector2d& p_pixel, double depth) const {
    // TODO:
    // 1. Unproject pixel to p_cam (using depth and K_inv)
    // 2. Transform p_cam to p_world
    return Eigen::Vector3d::Zero();
}
