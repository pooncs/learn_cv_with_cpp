#include "colorize.hpp"

std::vector<PointRGB> colorize_cloud(const std::vector<cv::Point3f>& cloud, const cv::Mat& rgb_img, const cv::Mat& K) {
    std::vector<PointRGB> result;
    result.reserve(cloud.size());
    
    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);
    
    for (const auto& p : cloud) {
        if (p.z <= 0) continue;
        
        int u = (int)(fx * p.x / p.z + cx);
        int v = (int)(fy * p.y / p.z + cy);
        
        if (u >= 0 && u < rgb_img.cols && v >= 0 && v < rgb_img.rows) {
            cv::Vec3b color = rgb_img.at<cv::Vec3b>(v, u);
            result.push_back({p.x, p.y, p.z, color[2], color[1], color[0]}); // BGR to RGB
        } else {
            // Keep point but make it black or remove? Let's keep it black.
            result.push_back({p.x, p.y, p.z, 0, 0, 0});
        }
    }
    
    return result;
}
