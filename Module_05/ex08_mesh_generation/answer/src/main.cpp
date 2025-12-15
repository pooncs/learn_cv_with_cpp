#include <iostream>
#include <cmath>
#include "mesh.hpp"

int main() {
    int W = 50, H = 50;
    std::vector<Point3D> vertices;
    vertices.reserve(W * H);
    
    // Generate a sine wave surface
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float z = std::sin(x * 0.2f) * std::cos(y * 0.2f);
            vertices.push_back({(float)x, (float)y, z * 5.0f});
        }
    }
    
    auto faces = triangulate_grid(W, H);
    write_obj("surface.obj", vertices, faces);
    
    std::cout << "Generated mesh with " << vertices.size() << " vertices and " << faces.size() << " triangles.\n";

    return 0;
}
