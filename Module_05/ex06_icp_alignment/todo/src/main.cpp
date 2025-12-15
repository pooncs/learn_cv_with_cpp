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
    
    // TODO: Call icp_align

    return 0;
}
