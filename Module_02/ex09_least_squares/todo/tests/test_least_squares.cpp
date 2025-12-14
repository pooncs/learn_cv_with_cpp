#include <iostream>
#include <cassert>
#include "least_squares.hpp"

void test_fit() {
    // Points on plane z=0: (0,0,0), (1,0,0), (0,1,0)
    Eigen::MatrixXd points(3, 3);
    points << 0, 1, 0,
              0, 0, 1,
              0, 0, 0;
    
    Eigen::Vector4d plane = fit_plane(points);
    // Should be [0, 0, 1, 0] or [0, 0, -1, 0]
    
    // Check if normal is (0,0,1)
    double dot = std::abs(plane(2));
    assert(std::abs(dot - 1.0) < 1e-4);
    assert(std::abs(plane(3)) < 1e-4); // d=0

    std::cout << "[PASS] fit_plane\n";
}

int main() {
    test_fit();
    return 0;
}
