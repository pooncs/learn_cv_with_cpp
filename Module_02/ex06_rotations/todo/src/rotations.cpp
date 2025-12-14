#include "rotations.hpp"

Eigen::Matrix3d euler_to_matrix(double roll, double pitch, double yaw) {
    // TODO: Convert Euler angles to Rotation Matrix
    // Convention: ZYX (Yaw, Pitch, Roll) -> R = Rz * Ry * Rx
    return Eigen::Matrix3d::Identity();
}

Eigen::AngleAxisd matrix_to_axis_angle(const Eigen::Matrix3d& R) {
    // TODO: Convert Matrix to AngleAxisd
    return Eigen::AngleAxisd(0, Eigen::Vector3d::UnitX());
}
