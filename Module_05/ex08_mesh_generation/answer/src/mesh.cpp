#include "mesh.hpp"
#include <fstream>
#include <stdexcept>

std::vector<Triangle> triangulate_grid(int width, int height) {
    std::vector<Triangle> faces;
    
    for (int y = 0; y < height - 1; ++y) {
        for (int x = 0; x < width - 1; ++x) {
            // Indices
            int i0 = y * width + x;
            int i1 = y * width + (x + 1);
            int i2 = (y + 1) * width + x;
            int i3 = (y + 1) * width + (x + 1);
            
            // Triangle 1: i0, i2, i1
            faces.push_back({i0, i2, i1});
            
            // Triangle 2: i1, i2, i3
            faces.push_back({i1, i2, i3});
        }
    }
    return faces;
}

void write_obj(const std::string& filename, const std::vector<Point3D>& vertices, const std::vector<Triangle>& faces) {
    std::ofstream out(filename);
    if (!out) throw std::runtime_error("Cannot open file");
    
    out << "# Generated Mesh\n";
    for (const auto& v : vertices) {
        out << "v " << v.x << " " << v.y << " " << v.z << "\n";
    }
    
    for (const auto& f : faces) {
        // OBJ indices are 1-based
        out << "f " << f.v0 + 1 << " " << f.v1 + 1 << " " << f.v2 + 1 << "\n";
    }
}
