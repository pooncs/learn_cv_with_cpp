#pragma once
#include <Eigen/Dense>

Eigen::Matrix2d extract_block(const Eigen::Matrix4d& M);
void set_row_zero(Eigen::Matrix4d& M, int row_idx);
void paste_block(Eigen::Matrix4d& M, const Eigen::Matrix2d& block);
