#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

struct Point3D {
    float x, y, z;
};

// TODO: Implement writeXYZ
// Should write each point as "x y z" on a new line
void writeXYZ(const std::string& filename, const std::vector<Point3D>& points) {
    // Your code here
}

// TODO: Implement readXYZ
// Should read "x y z" from each line
std::vector<Point3D> readXYZ(const std::string& filename) {
    std::vector<Point3D> points;
    // Your code here
    return points;
}

// TODO: Implement writePLY
// Should write a PLY header then the points
// Header format:
// ply
// format ascii 1.0
// element vertex <number_of_points>
// property float x
// property float y
// property float z
// end_header
void writePLY(const std::string& filename, const std::vector<Point3D>& points) {
    // Your code here
}

int main() {
    // Create some dummy data
    std::vector<Point3D> original_points = {
        {1.0f, 2.0f, 3.0f},
        {4.5f, 5.5f, 6.5f},
        {7.0f, 8.0f, 9.0f}
    };

    // 1. Write XYZ
    std::cout << "Writing XYZ file..." << std::endl;
    writeXYZ("test.xyz", original_points);

    // 2. Read XYZ
    std::cout << "Reading XYZ file..." << std::endl;
    auto loaded_points = readXYZ("test.xyz");

    // Check
    if (loaded_points.size() != original_points.size()) {
        std::cerr << "Error: Size mismatch!" << std::endl;
    } else {
        std::cout << "Read " << loaded_points.size() << " points successfully." << std::endl;
        // Basic check
        bool match = true;
        for (size_t i = 0; i < loaded_points.size(); ++i) {
            // Using a small epsilon for float comparison if needed, but direct assignment here
            if (loaded_points[i].x != original_points[i].x) {
                std::cerr << "Mismatch at index " << i << std::endl;
                match = false;
            }
        }
        if(match) std::cout << "Data verification passed." << std::endl;
    }

    // 3. Write PLY
    std::cout << "Writing PLY file..." << std::endl;
    writePLY("test.ply", original_points);
    std::cout << "Done. Check test.ply content." << std::endl;

    return 0;
}
