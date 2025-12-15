#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

struct Point3D {
    float x, y, z;
    uint8_t r, g, b;
};

class Camera {
public:
    Camera(int width, int height, float fov_deg = 60.0f);
    
    void lookAt(const Eigen::Vector3f& eye, const Eigen::Vector3f& center, const Eigen::Vector3f& up);
    
    Eigen::Matrix4f getViewMatrix() const { return view_; }
    Eigen::Matrix4f getProjectionMatrix() const { return proj_; }
    
    int width() const { return width_; }
    int height() const { return height_; }

private:
    Eigen::Matrix4f view_;
    Eigen::Matrix4f proj_;
    int width_, height_;
};

// Renders point cloud to image
cv::Mat render_point_cloud(const std::vector<Point3D>& cloud, const Camera& cam);
