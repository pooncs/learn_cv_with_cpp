#include "transformer.hpp"

Transformer::Transformer() {
    K = Eigen::Matrix3d::Identity();
    T_cw = Eigen::Matrix4d::Identity();
    T_wc = Eigen::Matrix4d::Identity();
}

void Transformer::set_intrinsics(double fx, double fy, double cx, double cy) {
    K << fx,  0, cx,
          0, fy, cy,
          0,  0,  1;
}

void Transformer::set_extrinsics(const Eigen::Matrix4d& T_cw) {
    this->T_cw = T_cw;
    this->T_wc = T_cw.inverse();
}

Eigen::Vector2d Transformer::project(const Eigen::Vector3d& p_world) const {
    // 1. Transform to Camera Frame
    Eigen::Vector4d p_w_h;
    p_w_h << p_world, 1.0;
    Eigen::Vector4d p_c_h = T_cw * p_w_h;
    Eigen::Vector3d p_c = p_c_h.head<3>();

    // Check Z > 0
    if (p_c(2) <= 1e-6) return Eigen::Vector2d(-1, -1); // Invalid

    // 2. Project
    Eigen::Vector3d p_pix_h = K * p_c;
    
    // 3. Normalize
    return Eigen::Vector2d(p_pix_h(0) / p_pix_h(2), p_pix_h(1) / p_pix_h(2));
}

Eigen::Vector3d Transformer::back_project(const Eigen::Vector2d& p_pixel, double depth) const {
    // 1. Un-project to Camera Frame
    // p_cam = Z * K_inv * [u, v, 1]
    Eigen::Vector3d p_pix_h(p_pixel(0), p_pixel(1), 1.0);
    Eigen::Vector3d p_c = depth * K.inverse() * p_pix_h;

    // 2. Transform to World Frame
    Eigen::Vector4d p_c_h;
    p_c_h << p_c, 1.0;
    Eigen::Vector4d p_w_h = T_wc * p_c_h;
    
    return p_w_h.head<3>();
}
