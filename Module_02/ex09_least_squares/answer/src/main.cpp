#include <iostream>
#include "least_squares.hpp"

int main() {
    // Generate points on plane z = 0 (so a=0, b=0, c=1, d=0)
    int N = 10;
    Eigen::MatrixXd points(3, N);
    for(int i=0; i<N; ++i) {
        points(0, i) = rand() % 10;
        points(1, i) = rand() % 10;
        points(2, i) = 0.0 + 0.01 * (rand() % 100 / 100.0); // Small noise
    }

    Eigen::Vector4d plane = fit_plane(points);
    std::cout << "Fitted Plane (a,b,c,d): " << plane.transpose() << "\n";
    std::cout << "Expected approx: 0, 0, 1, 0 (or flipped sign)\n";

    return 0;
}
