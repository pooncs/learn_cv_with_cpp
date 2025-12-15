#include "sor.hpp"

std::vector<Point3D> remove_outliers(const std::vector<Point3D>& cloud, int k, float std_mul) {
    // TODO:
    // 1. Calculate mean dist to k neighbors for each point
    // 2. Compute mean and stddev of these mean distances
    // 3. Keep points where dist <= mean + std_mul * stddev
    return {};
}
