#include <iostream>
#include "octree.hpp"

int main() {
    AABB bounds{{0, 0, 0}, {100, 100, 100}};
    Octree tree(bounds, 4);
    
    // Insert uniformly distributed points
    for(int i=0; i<100; ++i) {
        tree.insert({(float)i, (float)i, (float)i});
    }
    
    // Query near center
    Point3D center{50, 50, 50};
    float radius = 5.0f;
    
    auto points = tree.query_radius(center, radius);
    
    std::cout << "Found " << points.size() << " points within radius " << radius << "\n";
    for(auto& p : points) {
        std::cout << p.x << " " << p.y << " " << p.z << "\n";
    }

    return 0;
}
