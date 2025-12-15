#include "PointCloudIO.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>

std::vector<Point3D> read_xyz(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) throw std::runtime_error("Cannot open file");
    
    std::vector<Point3D> points;
    float x, y, z;
    while (in >> x >> y >> z) {
        points.push_back({x, y, z});
    }
    return points;
}

void write_xyz(const std::string& filename, const std::vector<Point3D>& points) {
    std::ofstream out(filename);
    if (!out) throw std::runtime_error("Cannot open file");
    
    for (const auto& p : points) {
        out << p.x << " " << p.y << " " << p.z << "\n";
    }
}

std::vector<Point3D> read_ply(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) throw std::runtime_error("Cannot open file");
    
    std::string line;
    int vertex_count = 0;
    bool header_ended = false;
    
    while (std::getline(in, line)) {
        if (line == "end_header") {
            header_ended = true;
            break;
        }
        if (line.rfind("element vertex", 0) == 0) {
            std::stringstream ss(line);
            std::string temp;
            ss >> temp >> temp >> vertex_count;
        }
    }
    
    if (!header_ended) throw std::runtime_error("Invalid PLY header");
    
    std::vector<Point3D> points;
    points.reserve(vertex_count);
    float x, y, z;
    while (in >> x >> y >> z) {
        points.push_back({x, y, z});
    }
    return points;
}

void write_ply(const std::string& filename, const std::vector<Point3D>& points) {
    std::ofstream out(filename);
    if (!out) throw std::runtime_error("Cannot open file");
    
    out << "ply\n";
    out << "format ascii 1.0\n";
    out << "element vertex " << points.size() << "\n";
    out << "property float x\n";
    out << "property float y\n";
    out << "property float z\n";
    out << "end_header\n";
    
    for (const auto& p : points) {
        out << p.x << " " << p.y << " " << p.z << "\n";
    }
}
