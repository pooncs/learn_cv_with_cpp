#include "quaternions.hpp"

Eigen::Quaterniond custom_slerp(const Eigen::Quaterniond& q1, const Eigen::Quaterniond& q2, double t) {
    // TODO: Implement SLERP
    // Hint: Eigen::Quaternion has a slerp method.
    return Eigen::Quaterniond::Identity();
}
