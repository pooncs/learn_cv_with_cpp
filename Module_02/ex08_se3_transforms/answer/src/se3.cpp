#include "se3.hpp"

Eigen::Matrix4d create_se3(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = R;
    T.block<3, 1>(0, 3) = t;
    return T;
}

Eigen::Vector3d apply_transform(const Eigen::Matrix4d& T, const Eigen::Vector3d& p) {
    Eigen::Vector4d ph;
    ph << p, 1.0;
    Eigen::Vector4d p_transformed = T * ph;
    return p_transformed.head<3>();
}
