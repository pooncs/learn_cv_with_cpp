#include "PointCloudIO.hpp"
#include <fstream>

std::vector<Point3D> read_xyz(const std::string& filename) {
    // TODO: Read X Y Z from file
    return {};
}

void write_xyz(const std::string& filename, const std::vector<Point3D>& points) {
    // TODO: Write X Y Z to file
}

std::vector<Point3D> read_ply(const std::string& filename) {
    // TODO: Parse PLY header, read element vertex count, read data
    return {};
}

void write_ply(const std::string& filename, const std::vector<Point3D>& points) {
    // TODO: Write PLY header, write data
}
