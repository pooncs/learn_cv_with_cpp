#include <iostream>
#include "normals.hpp"

int main() {
    // Create a plane z=0
    std::vector<Point3D> cloud;
    for(int x=0; x<10; ++x) {
        for(int y=0; y<10; ++y) {
            cloud.push_back({(float)x, (float)y, 0.0f});
        }
    }
    
    // Add a bit of noise to z
    cloud[55].z = 0.1f; 
    
    auto normals = compute_normals(cloud, 5);
    
    std::cout << "Computed " << normals.size() << " normals.\n";
    // Check center point normal, should be close to (0,0,1)
    auto n = normals[55];
    std::cout << "Center normal: " << n.nx << " " << n.ny << " " << n.nz << "\n";

    return 0;
}
