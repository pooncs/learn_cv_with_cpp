#include <iostream>
#include <cassert>
#include "block_ops.hpp"

void test_extract() {
    Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
    // Identity:
    // 1 0 0 0
    // 0 1 0 0
    // 0 0 1 0
    // 0 0 0 1
    // Block(1,1) is:
    // 1 0
    // 0 1
    Eigen::Matrix2d b = extract_block(M);
    assert(b(0,0) == 1 && b(1,1) == 1);
    assert(b(0,1) == 0 && b(1,0) == 0);
    std::cout << "[PASS] extract_block\n";
}

void test_set_row() {
    Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
    set_row_zero(M, 0);
    assert(M(0,0) == 0);
    assert(M(1,1) == 1); // others untouched
    std::cout << "[PASS] set_row_zero\n";
}

void test_paste() {
    Eigen::Matrix4d M = Eigen::Matrix4d::Zero();
    Eigen::Matrix2d ones = Eigen::Matrix2d::Constant(1.0);
    paste_block(M, ones);
    // Bottom right 2x2 should be 1
    assert(M(2,2) == 1 && M(3,3) == 1);
    assert(M(0,0) == 0);
    std::cout << "[PASS] paste_block\n";
}

int main() {
    test_extract();
    test_set_row();
    test_paste();
    return 0;
}
