#include "mat_info.hpp"
#include <sstream>

std::string get_mat_info(const cv::Mat& m) {
    std::stringstream ss;
    ss << "Dims: " << m.rows << "x" << m.cols << "\n";
    ss << "Channels: " << m.channels() << "\n";
    ss << "Depth: " << m.depth() << "\n";
    ss << "ElemSize: " << m.elemSize() << " bytes\n";
    ss << "Step: " << m.step << " bytes\n";
    ss << "IsContinuous: " << (m.isContinuous() ? "Yes" : "No") << "\n";
    return ss.str();
}

bool share_data(const cv::Mat& m1, const cv::Mat& m2) {
    return m1.data == m2.data;
}
