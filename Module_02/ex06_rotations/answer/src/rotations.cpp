#include "rotations.hpp"

Eigen::Matrix3d euler_to_matrix(double roll, double pitch, double yaw) {
    // ZYX order: R = Rz * Ry * Rx
    Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());

    Eigen::Quaterniond q = yawAngle * pitchAngle * rollAngle;
    return q.matrix();
}

Eigen::AngleAxisd matrix_to_axis_angle(const Eigen::Matrix3d& R) {
    return Eigen::AngleAxisd(R);
}
