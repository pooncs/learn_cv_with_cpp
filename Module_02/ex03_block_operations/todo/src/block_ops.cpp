#include "block_ops.hpp"

Eigen::Matrix2d extract_block(const Eigen::Matrix4d& M) {
    // TODO: Extract 2x2 block starting at (1,1)
    return Eigen::Matrix2d::Zero();
}

void set_row_zero(Eigen::Matrix4d& M, int row_idx) {
    // TODO: Set the row at row_idx to zero
}

void paste_block(Eigen::Matrix4d& M, const Eigen::Matrix2d& block) {
    // TODO: Paste 'block' into bottom-right corner of 'M' (starts at 2,2)
}
