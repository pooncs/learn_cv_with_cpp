#pragma once
#include <vector>

struct Point3D {
    float x, y, z;
};

std::vector<Point3D> remove_outliers(const std::vector<Point3D>& cloud, int k, float std_mul);
