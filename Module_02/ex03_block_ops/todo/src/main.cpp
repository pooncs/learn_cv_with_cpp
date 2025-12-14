#include <iostream>
#include "block_ops.hpp"

int main() {
    Eigen::Matrix4d M;
    M << 1,  2,  3,  4,
         5,  6,  7,  8,
         9, 10, 11, 12,
        13, 14, 15, 16;
    
    std::cout << "Original Matrix:\n" << M << "\n\n";

    std::cout << "=== Task 1: Extract Block ===\n";
    Eigen::Matrix2d b = extract_block(M);
    std::cout << "Extracted block (1,1):\n" << b << "\n\n";

    std::cout << "=== Task 2: Set Row Zero ===\n";
    set_row_zero(M, 2);
    std::cout << "Matrix after row 2 set to zero:\n" << M << "\n\n";

    std::cout << "=== Task 3: Paste Block ===\n";
    Eigen::Matrix2d ones = Eigen::Matrix2d::Constant(1.0);
    paste_block(M, ones);
    std::cout << "Matrix after pasting ones to bottom-right:\n" << M << "\n";

    return 0;
}
