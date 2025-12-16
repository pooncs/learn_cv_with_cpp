#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

struct Point3D {
    float x, y, z;
};

// Implement writeXYZ
void writeXYZ(const std::string& filename, const std::vector<Point3D>& points) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }
    for (const auto& p : points) {
        outfile << p.x << " " << p.y << " " << p.z << "\n";
    }
    outfile.close();
}

// Implement readXYZ
std::vector<Point3D> readXYZ(const std::string& filename) {
    std::vector<Point3D> points;
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error opening file for reading: " << filename << std::endl;
        return points;
    }

    std::string line;
    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        Point3D p;
        if (ss >> p.x >> p.y >> p.z) {
            points.push_back(p);
        }
    }
    infile.close();
    return points;
}

// Implement writePLY
void writePLY(const std::string& filename, const std::vector<Point3D>& points) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    // Write Header
    outfile << "ply\n";
    outfile << "format ascii 1.0\n";
    outfile << "element vertex " << points.size() << "\n";
    outfile << "property float x\n";
    outfile << "property float y\n";
    outfile << "property float z\n";
    outfile << "end_header\n";

    // Write Data
    for (const auto& p : points) {
        outfile << p.x << " " << p.y << " " << p.z << "\n";
    }
    outfile.close();
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
