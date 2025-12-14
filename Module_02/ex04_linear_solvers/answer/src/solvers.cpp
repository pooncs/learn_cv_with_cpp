#include "solvers.hpp"

Eigen::VectorXd solve_llt(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
    return A.llt().solve(b);
}

Eigen::VectorXd solve_ldlt(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
    return A.ldlt().solve(b);
}

double relative_error(const Eigen::MatrixXd& A, const Eigen::VectorXd& x, const Eigen::VectorXd& b) {
    return (A * x - b).norm() / b.norm();
}
