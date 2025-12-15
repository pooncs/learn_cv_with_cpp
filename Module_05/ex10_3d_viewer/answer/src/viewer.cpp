#include "viewer.hpp"
#include <cmath>
#include <algorithm>

Camera::Camera(int width, int height, float fov_deg) : width_(width), height_(height) {
    float aspect = (float)width / height;
    float tan_half_fov = std::tan(fov_deg * 0.5f * M_PI / 180.0f);
    float z_near = 0.1f;
    float z_far = 1000.0f;
    
    proj_ = Eigen::Matrix4f::Zero();
    proj_(0, 0) = 1.0f / (aspect * tan_half_fov);
    proj_(1, 1) = 1.0f / tan_half_fov;
    proj_(2, 2) = -(z_far + z_near) / (z_far - z_near);
    proj_(2, 3) = -(2.0f * z_far * z_near) / (z_far - z_near);
    proj_(3, 2) = -1.0f;
}

void Camera::lookAt(const Eigen::Vector3f& eye, const Eigen::Vector3f& center, const Eigen::Vector3f& up) {
    Eigen::Vector3f f = (center - eye).normalized();
    Eigen::Vector3f s = f.cross(up).normalized();
    Eigen::Vector3f u = s.cross(f);
    
    view_ = Eigen::Matrix4f::Identity();
    view_(0, 0) = s.x(); view_(0, 1) = s.y(); view_(0, 2) = s.z();
    view_(1, 0) = u.x(); view_(1, 1) = u.y(); view_(1, 2) = u.z();
    view_(2, 0) = -f.x(); view_(2, 1) = -f.y(); view_(2, 2) = -f.z();
    view_(0, 3) = -s.dot(eye);
    view_(1, 3) = -u.dot(eye);
    view_(2, 3) = f.dot(eye);
}

cv::Mat render_point_cloud(const std::vector<Point3D>& cloud, const Camera& cam) {
    cv::Mat img = cv::Mat::zeros(cam.height(), cam.width(), CV_8UC3);
    
    Eigen::Matrix4f VP = cam.getProjectionMatrix() * cam.getViewMatrix();
    
    // Sort by depth (simple Z-buffering/Painter's algorithm)
    // Actually, we should project first then sort by Z (NDC)
    
    struct ProjectPt {
        int u, v;
        float z;
        uint8_t r, g, b;
    };
    
    std::vector<ProjectPt> projected;
    projected.reserve(cloud.size());
    
    for (const auto& p : cloud) {
        Eigen::Vector4f v(p.x, p.y, p.z, 1.0f);
        Eigen::Vector4f clip = VP * v;
        
        if (clip.w() <= 0) continue; // Behind camera
        
        // Perspective divide
        float ndc_x = clip.x() / clip.w();
        float ndc_y = clip.y() / clip.w();
        float ndc_z = clip.z() / clip.w();
        
        if (ndc_z < -1.0f || ndc_z > 1.0f) continue;
        
        // Viewport
        int u = (int)((ndc_x + 1.0f) * 0.5f * cam.width());
        int v_coord = (int)((1.0f - ndc_y) * 0.5f * cam.height()); // Flip Y
        
        if (u >= 0 && u < cam.width() && v_coord >= 0 && v_coord < cam.height()) {
            projected.push_back({u, v_coord, ndc_z, p.r, p.g, p.b});
        }
    }
    
    // Sort back-to-front (high Z to low Z in NDC? NDC Z maps near -1, far 1. So draw far first (1) -> near (-1))
    // Wait, OpenGL default is -1 near, 1 far? Or -z view? 
    // Projection matrix above maps near to -1, far to 1.
    // So we want to draw 1 first, then -1.
    std::sort(projected.begin(), projected.end(), [](const ProjectPt& a, const ProjectPt& b){
        return a.z > b.z;
    });
    
    for (const auto& p : projected) {
        // Draw small circle or point
        cv::circle(img, cv::Point(p.u, p.v), 1, cv::Scalar(p.b, p.g, p.r), -1);
    }
    
    return img;
}
