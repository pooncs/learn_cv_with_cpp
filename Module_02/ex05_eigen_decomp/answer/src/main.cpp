#include <iostream>
#include "eigen_decomp.hpp"

int main() {
    // Generate synthetic data: Points along a line y = x with some noise
    int N = 100;
    Eigen::MatrixXd points(2, N);
    for(int i=0; i<N; ++i) {
        double t = (double)i / N; // 0 to 1
        points(0, i) = t + 0.1 * (rand() / (double)RAND_MAX);
        points(1, i) = t + 0.1 * (rand() / (double)RAND_MAX);
    }

    auto result = compute_pca(points);
    Eigen::Vector2d mean = result.first;
    Eigen::Matrix2d eigenvectors = result.second;

    std::cout << "Mean:\n" << mean << "\n\n";
    std::cout << "Eigenvectors (cols):\n" << eigenvectors << "\n\n";
    
    // The principal axis (largest eigenvalue) should be the second column (Eigen sorts increasing)
    // For y=x line, direction is [1, 1] normalized -> [0.707, 0.707]
    std::cout << "Principal Axis:\n" << eigenvectors.col(1) << "\n";

    return 0;
}
