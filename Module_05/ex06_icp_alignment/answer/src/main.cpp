#include <iostream>
#include "icp.hpp"

int main() {
    std::vector<Point3D> target = {
        {0, 0, 0}, {1, 0, 0}, {0, 1, 0}
    };
    
    // Source is shifted by (0.1, 0.1, 0)
    std::vector<Point3D> source = {
        {0.1f, 0.1f, 0}, {1.1f, 0.1f, 0}, {0.1f, 1.1f, 0}
    };
    
    Eigen::Matrix4f T = icp_align(source, target, 5);
    
    std::cout << "Estimated Transform:\n" << T << "\n";
    // Expected t ~ [-0.1, -0.1, 0]

    return 0;
}
