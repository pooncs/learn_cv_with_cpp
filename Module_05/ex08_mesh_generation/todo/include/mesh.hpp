#pragma once
#include <vector>
#include <string>

struct Point3D {
    float x, y, z;
};

struct Triangle {
    int v0, v1, v2;
};

std::vector<Triangle> triangulate_grid(int width, int height);
void write_obj(const std::string& filename, const std::vector<Point3D>& vertices, const std::vector<Triangle>& faces);
