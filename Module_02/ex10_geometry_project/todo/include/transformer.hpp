#pragma once
#include <Eigen/Dense>

class Transformer {
public:
    Transformer();

    // Set Intrinsics (fx, fy, cx, cy)
    void set_intrinsics(double fx, double fy, double cx, double cy);

    // Set Extrinsics (World to Camera)
    void set_extrinsics(const Eigen::Matrix4d& T_cw);

    // Project 3D world point to 2D pixel
    Eigen::Vector2d project(const Eigen::Vector3d& p_world) const;

    // Back-project 2D pixel with depth to 3D world point
    Eigen::Vector3d back_project(const Eigen::Vector2d& p_pixel, double depth) const;

private:
    Eigen::Matrix3d K;
    Eigen::Matrix4d T_cw; // World -> Camera
    Eigen::Matrix4d T_wc; // Camera -> World (Inverse)
};
