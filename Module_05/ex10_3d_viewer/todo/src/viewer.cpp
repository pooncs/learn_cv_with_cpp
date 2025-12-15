#include "viewer.hpp"

Camera::Camera(int width, int height, float fov_deg) : width_(width), height_(height) {
    // TODO: Compute projection matrix
    proj_ = Eigen::Matrix4f::Identity();
}

void Camera::lookAt(const Eigen::Vector3f& eye, const Eigen::Vector3f& center, const Eigen::Vector3f& up) {
    // TODO: Compute View Matrix (Gram-Schmidt)
    // f = normalize(center - eye)
    // s = normalize(cross(f, up))
    // u = cross(s, f)
    view_ = Eigen::Matrix4f::Identity();
}

Eigen::Matrix4f Camera::getViewMatrix() const { return view_; }
Eigen::Matrix4f Camera::getProjectionMatrix() const { return proj_; }
int Camera::width() const { return width_; }
int Camera::height() const { return height_; }

cv::Mat render_point_cloud(const std::vector<Point3D>& cloud, const Camera& cam) {
    // TODO:
    // 1. VP = Proj * View
    // 2. For each point p:
    //    clip = VP * p
    //    ndc = clip / clip.w
    //    viewport transform -> u, v
    //    draw
    return cv::Mat();
}
