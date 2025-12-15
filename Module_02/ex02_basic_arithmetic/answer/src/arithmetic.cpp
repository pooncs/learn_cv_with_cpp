#include "arithmetic.hpp"

Eigen::Matrix2d mat_mul(const Eigen::Matrix2d& A, const Eigen::Matrix2d& B) {
    return A * B;
}

Eigen::Matrix2d element_wise(const Eigen::Matrix2d& A, const Eigen::Matrix2d& B) {
    return A.array() * B.array();
}

void broadcast_add(Eigen::MatrixXd& M, const Eigen::VectorXd& v) {
    M.colwise() += v;
}
