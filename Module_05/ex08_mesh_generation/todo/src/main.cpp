#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <fmt/core.h>

struct Point3D {
    float x, y, z;
};

struct Triangle {
    int v0, v1, v2; // Indices into point list
};

// TODO: Implement generateMesh
// Given a grid of points (width x height), generate triangles
std::vector<Triangle> generateMesh(int width, int height) {
    std::vector<Triangle> mesh;
    // Iterate over grid cells (y, x) up to (height-1, width-1)
    // For each cell, create 2 triangles:
    // T1: (x, y), (x, y+1), (x+1, y)
    // T2: (x+1, y), (x, y+1), (x+1, y+1)
    
    // Note: Indices are y * width + x
    
    return mesh;
}

void writeOBJ(const std::string& filename, const std::vector<Point3D>& points, const std::vector<Triangle>& mesh) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) return;

    outfile << "# Simple Mesh\n";
    for (const auto& p : points) {
        outfile << "v " << p.x << " " << p.y << " " << p.z << "\n";
    }

    for (const auto& t : mesh) {
        // OBJ indices are 1-based
        outfile << "f " << t.v0 + 1 << " " << t.v1 + 1 << " " << t.v2 + 1 << "\n";
    }
    outfile.close();
}

int main() {
    int width = 10;
    int height = 10;
    std::vector<Point3D> points;

    // 1. Create a grid of points (e.g., a hill shape)
    for(int y=0; y<height; ++y) {
        for(int x=0; x<width; ++x) {
            float z = std::sin(x * 0.5f) * std::cos(y * 0.5f);
            points.push_back({(float)x, (float)y, z});
        }
    }
    fmt::print("Created {} points.\n", points.size());

    // 2. Generate Mesh
    auto mesh = generateMesh(width, height);
    fmt::print("Generated {} triangles.\n", mesh.size());

    // 3. Save
    writeOBJ("mesh.obj", points, mesh);
    fmt::print("Saved mesh.obj\n");

    return 0;
}
