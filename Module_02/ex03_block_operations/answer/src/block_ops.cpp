#include "block_ops.hpp"

Eigen::Matrix2d extract_block(const Eigen::Matrix4d& M) {
    // Extract 2x2 block starting at (1,1)
    return M.block<2, 2>(1, 1);
}

void set_row_zero(Eigen::Matrix4d& M, int row_idx) {
    if(row_idx >= 0 && row_idx < 4) {
        M.row(row_idx).setZero();
    }
}

void paste_block(Eigen::Matrix4d& M, const Eigen::Matrix2d& block) {
    // Paste into bottom-right corner (starts at 2,2 for 4x4 matrix and 2x2 block)
    M.block<2, 2>(2, 2) = block;
}
