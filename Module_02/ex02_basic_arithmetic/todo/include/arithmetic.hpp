#pragma once
#include <Eigen/Dense>

Eigen::Matrix2d mat_mul(const Eigen::Matrix2d& A, const Eigen::Matrix2d& B);
Eigen::Matrix2d element_wise(const Eigen::Matrix2d& A, const Eigen::Matrix2d& B);
void broadcast_add(Eigen::MatrixXd& M, const Eigen::VectorXd& v);
