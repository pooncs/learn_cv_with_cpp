#include "se3.hpp"

Eigen::Matrix4d create_se3(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
    // TODO: Construct 4x4 matrix
    // [ R t ]
    // [ 0 1 ]
    return Eigen::Matrix4d::Identity();
}

Eigen::Vector3d apply_transform(const Eigen::Matrix4d& T, const Eigen::Vector3d& p) {
    // TODO: Convert p to homogeneous [x,y,z,1], multiply by T, convert back.
    return p;
}
