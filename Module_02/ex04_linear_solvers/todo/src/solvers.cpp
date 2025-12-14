#include "solvers.hpp"

Eigen::VectorXd solve_llt(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
    // TODO: Solve using A.llt()
    return Eigen::VectorXd::Zero(b.size());
}

Eigen::VectorXd solve_ldlt(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
    // TODO: Solve using A.ldlt()
    return Eigen::VectorXd::Zero(b.size());
}

double relative_error(const Eigen::MatrixXd& A, const Eigen::VectorXd& x, const Eigen::VectorXd& b) {
    // TODO: Compute norm(A*x - b) / norm(b)
    return 0.0;
}
