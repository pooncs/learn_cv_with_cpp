#pragma once
#include <string>
#include <vector>

struct Point3D {
    float x, y, z;
};

std::vector<Point3D> read_xyz(const std::string& filename);
void write_xyz(const std::string& filename, const std::vector<Point3D>& points);

std::vector<Point3D> read_ply(const std::string& filename);
void write_ply(const std::string& filename, const std::vector<Point3D>& points);
