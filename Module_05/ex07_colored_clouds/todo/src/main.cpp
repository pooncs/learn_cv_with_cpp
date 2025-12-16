#include <iostream>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <fmt/core.h>

struct PointRGB {
    float x, y, z;
    uint8_t r, g, b;
};

struct CameraIntrinsics {
    float fx, fy;
    float cx, cy;
};

// TODO: Implement mapColorToCloud
// Projects each point to image plane and assigns color
void mapColorToCloud(std::vector<PointRGB>& cloud, const cv::Mat& image, const CameraIntrinsics& K) {
    // For each point:
    // u = fx * X / Z + cx
    // v = fy * Y / Z + cy
    // Check bounds (0 <= u < width, 0 <= v < height)
    // Sample image at (u, v)
    // Assign r,g,b
    
    // Note: OpenCV uses BGR by default
}

void writePLY(const std::string& filename, const std::vector<PointRGB>& points) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) return;

    outfile << "ply\n";
    outfile << "format ascii 1.0\n";
    outfile << "element vertex " << points.size() << "\n";
    outfile << "property float x\n";
    outfile << "property float y\n";
    outfile << "property float z\n";
    outfile << "property uchar red\n";
    outfile << "property uchar green\n";
    outfile << "property uchar blue\n";
    outfile << "end_header\n";

    for (const auto& p : points) {
        outfile << p.x << " " << p.y << " " << p.z << " " 
                << (int)p.r << " " << (int)p.g << " " << (int)p.b << "\n";
    }
    outfile.close();
}

int main() {
    // 1. Create Intrinsics
    int width = 640;
    int height = 480;
    CameraIntrinsics K = {500.0f, 500.0f, 320.0f, 240.0f};

    // 2. Create synthetic image (Checkerboard)
    cv::Mat image(height, width, CV_8UC3);
    int check_size = 50;
    for(int y=0; y<height; ++y) {
        for(int x=0; x<width; ++x) {
            bool black = ((x/check_size) + (y/check_size)) % 2 == 0;
            image.at<cv::Vec3b>(y, x) = black ? cv::Vec3b(0,0,0) : cv::Vec3b(255,255,255);
        }
    }
    
    // 3. Create synthetic cloud (Plane at Z=2.0)
    std::vector<PointRGB> cloud;
    for(int v=0; v<height; v+=5) { // Subsample
        for(int u=0; u<width; u+=5) {
             float Z = 2.0f;
             float X = (u - K.cx) * Z / K.fx;
             float Y = (v - K.cy) * Z / K.fy;
             cloud.push_back({X, Y, Z, 0, 0, 0});
        }
    }
    fmt::print("Created cloud with {} points.\n", cloud.size());

    // 4. Map Color
    mapColorToCloud(cloud, image, K);

    // 5. Save
    writePLY("colored.ply", cloud);
    fmt::print("Saved colored.ply\n");

    return 0;
}
