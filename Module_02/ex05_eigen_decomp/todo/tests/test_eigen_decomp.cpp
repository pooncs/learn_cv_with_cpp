#include <iostream>
#include <cassert>
#include "eigen_decomp.hpp"

void test_pca() {
    // Create points perfectly along X axis
    // (-1, 0), (0, 0), (1, 0)
    Eigen::MatrixXd points(2, 3);
    points << -1, 0, 1,
               0, 0, 0;
    
    auto result = compute_pca(points);
    Eigen::Vector2d mean = result.first;
    Eigen::Matrix2d vecs = result.second;

    // Mean should be (0, 0)
    assert(std::abs(mean(0)) < 1e-6);
    assert(std::abs(mean(1)) < 1e-6);

    // Principal axis (largest eigenvalue) should be X axis (1, 0) or (-1, 0)
    // Eigen sorts eigenvalues ascending, so last col is principal
    Eigen::Vector2d principal = vecs.col(1);
    
    // Check if principal is aligned with X axis
    assert(std::abs(std::abs(principal(0)) - 1.0) < 1e-6);
    assert(std::abs(principal(1)) < 1e-6);

    std::cout << "[PASS] compute_pca\n";
}

int main() {
    test_pca();
    return 0;
}
