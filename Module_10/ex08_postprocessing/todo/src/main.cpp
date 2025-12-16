#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

struct Detection {
    cv::Rect box;
    float conf;
    int classId;
};

// TODO: Implement postprocess function
// void postprocess(float* data, int rows, int dimensions, float confThreshold, std::vector<Detection>& output)

int main() {
    std::cout << "Post-processing Exercise" << std::endl;

    // TODO: Create dummy data

    // TODO: Call postprocess

    // TODO: Verify results

    return 0;
}
