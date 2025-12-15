#include <iostream>
#include "PointCloudIO.hpp"

int main() {
    std::vector<Point3D> points = {
        {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}
    };
    
    write_xyz("test.xyz", points);
    auto p1 = read_xyz("test.xyz");
    std::cout << "Read XYZ: " << p1.size() << " points.\n";
    
    write_ply("test.ply", points);
    auto p2 = read_ply("test.ply");
    std::cout << "Read PLY: " << p2.size() << " points.\n";

    return 0;
}
