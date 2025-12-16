#include <iostream>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <fmt/core.h>

struct Point3D {
    float x, y, z;
};

struct CameraIntrinsics {
    float fx, fy;
    float cx, cy;
};

// Implement depthToCloud
std::vector<Point3D> depthToCloud(const cv::Mat& depth_img, const CameraIntrinsics& K) {
    std::vector<Point3D> cloud;
    // Iterate over rows (v) and cols (u)
    for (int v = 0; v < depth_img.rows; ++v) {
        for (int u = 0; u < depth_img.cols; ++u) {
            float Z = depth_img.at<float>(v, u);
            
            // Skip invalid depth if necessary (e.g. Z <= 0 or NaN)
            if (Z <= 0.0f) continue;

            float X = (u - K.cx) * Z / K.fx;
            float Y = (v - K.cy) * Z / K.fy;
            
            cloud.push_back({X, Y, Z});
        }
    }
    return cloud;
}

void writeXYZ(const std::string& filename, const std::vector<Point3D>& points) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) return;
    for (const auto& p : points) {
        outfile << p.x << " " << p.y << " " << p.z << "\n";
    }
    outfile.close();
}

int main() {
    // 1. Create a synthetic depth image (e.g., 640x480)
    int width = 640;
    int height = 480;
    cv::Mat depth_img = cv::Mat::zeros(height, width, CV_32F);

    // Fill with a gradient or simple shape (e.g., a plane)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Example: Plane sloped in Z
            depth_img.at<float>(y, x) = 2.0f + (float)x / width; // Z ranges from 2.0 to 3.0
        }
    }

    // 2. Define Intrinsics (Typical VGA camera)
    CameraIntrinsics K;
    K.fx = 500.0f;
    K.fy = 500.0f;
    K.cx = width / 2.0f;
    K.cy = height / 2.0f;

    // 3. Convert
    fmt::print("Converting depth map to point cloud...\n");
    auto cloud = depthToCloud(depth_img, K);

    // 4. Save
    fmt::print("Generated {} points. Saving to output.xyz...\n", cloud.size());
    writeXYZ("output.xyz", cloud);
    
    fmt::print("Done.\n");

    return 0;
}
