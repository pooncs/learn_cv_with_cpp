#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

struct Detection {
    cv::Rect box;
    float conf;
    int classId;
};

// TODO: Implement postprocess
// std::vector<Detection> postprocess(float* data, int num_boxes, int num_classes, float conf_threshold)

int main() {
    // TODO: Create dummy data
    
    // TODO: Call postprocess
    
    // TODO: Print detections
    
    return 0;
}
