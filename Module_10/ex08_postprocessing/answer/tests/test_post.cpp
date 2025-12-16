#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cassert>

struct Detection {
    cv::Rect box;
    float conf;
    int classId;
};

// Copy of the logic from main.cpp for testing
void postprocess(float* data, int rows, int dimensions, float confThreshold, std::vector<Detection>& output) {
    for (int i = 0; i < rows; ++i) {
        float* sample = data + i * dimensions;

        float* classesScores = sample + 4;
        int numClasses = dimensions - 4;
        
        int bestClassId = -1;
        float bestConf = -1.0f;

        for (int c = 0; c < numClasses; ++c) {
            if (classesScores[c] > bestConf) {
                bestConf = classesScores[c];
                bestClassId = c;
            }
        }

        if (bestConf > confThreshold) {
            float cx = sample[0];
            float cy = sample[1];
            float w = sample[2];
            float h = sample[3];

            int left = static_cast<int>(cx - w * 0.5f);
            int top = static_cast<int>(cy - h * 0.5f);
            int width = static_cast<int>(w);
            int height = static_cast<int>(h);

            output.push_back({cv::Rect(left, top, width, height), bestConf, bestClassId});
        }
    }
}

int main() {
    const int rows = 5; 
    const int numClasses = 2;
    const int dimensions = 4 + numClasses; // 6
    
    std::vector<float> data(rows * dimensions, 0.0f);

    // Row 2: Valid
    int targetIdx = 2;
    data[targetIdx * dimensions + 0] = 100.0f;
    data[targetIdx * dimensions + 1] = 100.0f;
    data[targetIdx * dimensions + 2] = 50.0f;
    data[targetIdx * dimensions + 3] = 50.0f;
    data[targetIdx * dimensions + 4] = 0.1f;
    data[targetIdx * dimensions + 5] = 0.9f;

    std::vector<Detection> dets;
    postprocess(data.data(), rows, dimensions, 0.5f, dets);

    if (dets.size() != 1) {
        std::cerr << "Expected 1 detection, got " << dets.size() << std::endl;
        return 1;
    }

    if (dets[0].classId != 1) {
        std::cerr << "Expected classId 1, got " << dets[0].classId << std::endl;
        return 1;
    }
    
    // Check Box
    // cx=100, w=50 -> left = 100 - 25 = 75
    // cy=100, h=50 -> top = 100 - 25 = 75
    if (dets[0].box.x != 75 || dets[0].box.y != 75) {
        std::cerr << "Box coordinates incorrect: " << dets[0].box << std::endl;
        return 1;
    }

    std::cout << "Post-processing Tests Passed" << std::endl;
    return 0;
}
