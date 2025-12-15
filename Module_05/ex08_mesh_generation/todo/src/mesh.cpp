#include "mesh.hpp"
#include <fstream>

std::vector<Triangle> triangulate_grid(int width, int height) {
    // TODO:
    // Generate 2 triangles for each quad (x, y) -> (x+1, y) -> (x, y+1) -> (x+1, y+1)
    return {};
}

void write_obj(const std::string& filename, const std::vector<Point3D>& vertices, const std::vector<Triangle>& faces) {
    // TODO: Write OBJ format
    // v x y z
    // f i j k (1-based indices)
}
