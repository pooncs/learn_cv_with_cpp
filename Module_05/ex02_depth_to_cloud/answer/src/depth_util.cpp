#include "depth_util.hpp"

std::vector<Point3D> depth_to_cloud(const cv::Mat& depth, const cv::Mat& K, float depth_scale) {
    std::vector<Point3D> cloud;
    
    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);
    
    for (int v = 0; v < depth.rows; ++v) {
        for (int u = 0; u < depth.cols; ++u) {
            float z = 0.0f;
            
            if (depth.type() == CV_16U) {
                z = depth.at<uint16_t>(v, u) * depth_scale;
            } else if (depth.type() == CV_32F) {
                z = depth.at<float>(v, u) * depth_scale;
            }
            
            if (z <= 0 || std::isnan(z)) continue;
            
            float x = (u - cx) * z / fx;
            float y = (v - cy) * z / fy;
            
            cloud.push_back({x, y, z});
        }
    }
    return cloud;
}
