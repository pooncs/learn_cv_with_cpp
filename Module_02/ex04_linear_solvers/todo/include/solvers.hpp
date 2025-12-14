#pragma once
#include <Eigen/Dense>

// Solves Ax = b using LLT decomposition
Eigen::VectorXd solve_llt(const Eigen::MatrixXd& A, const Eigen::VectorXd& b);

// Solves Ax = b using LDLT decomposition
Eigen::VectorXd solve_ldlt(const Eigen::MatrixXd& A, const Eigen::VectorXd& b);

// Computes relative error ||Ax - b|| / ||b||
double relative_error(const Eigen::MatrixXd& A, const Eigen::VectorXd& x, const Eigen::VectorXd& b);
