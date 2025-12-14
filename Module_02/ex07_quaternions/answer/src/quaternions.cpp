#include "quaternions.hpp"

Eigen::Quaterniond custom_slerp(const Eigen::Quaterniond& q1, const Eigen::Quaterniond& q2, double t) {
    // Eigen has built-in slerp:
    return q1.slerp(t, q2);
}
