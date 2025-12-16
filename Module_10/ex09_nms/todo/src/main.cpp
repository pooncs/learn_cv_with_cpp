#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

struct Detection {
    cv::Rect box;
    float conf;
    int classId;
};

// TODO: Implement IoU
// float iou(const cv::Rect& a, const cv::Rect& b)

// TODO: Implement NMS
// std::vector<Detection> nms(const std::vector<Detection>& dets, float threshold)

int main() {
    // TODO: Create overlapping detections
    
    // TODO: Run NMS
    
    // TODO: Print results
    
    return 0;
}
