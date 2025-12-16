#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>

struct Detection {
    cv::Rect box;
    float conf;
    int classId;
};

// Simulate YOLOv8 output: [1, 84, 8400] -> flattened
// For simplicity in C++, we often prefer [8400, 84] (Rows = Anchors, Cols = Features)
// Features: cx, cy, w, h, class0_score, class1_score, ...
void postprocess(float* data, int rows, int dimensions, float confThreshold, std::vector<Detection>& output) {
    for (int i = 0; i < rows; ++i) {
        float* sample = data + i * dimensions;

        // Find best class score
        // YOLOv8: first 4 are box, rest are classes
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
    std::cout << "Post-processing Demo" << std::endl;

    const int rows = 5; // Small number for demo
    const int numClasses = 2;
    const int dimensions = 4 + numClasses; // 6
    
    // Create dummy data: 5 rows (anchors), 6 columns (features)
    std::vector<float> data(rows * dimensions, 0.0f);

    // Row 2: A valid detection
    // cx=100, cy=100, w=50, h=50, class0=0.1, class1=0.9
    int targetIdx = 2;
    data[targetIdx * dimensions + 0] = 100.0f;
    data[targetIdx * dimensions + 1] = 100.0f;
    data[targetIdx * dimensions + 2] = 50.0f;
    data[targetIdx * dimensions + 3] = 50.0f;
    data[targetIdx * dimensions + 4] = 0.1f;
    data[targetIdx * dimensions + 5] = 0.9f;

    std::vector<Detection> dets;
    postprocess(data.data(), rows, dimensions, 0.5f, dets);

    std::cout << "Found " << dets.size() << " detections." << std::endl;
    for (const auto& d : dets) {
        std::cout << "Box: " << d.box << ", Conf: " << d.conf << ", Class: " << d.classId << std::endl;
    }

    if (dets.size() == 1 && dets[0].classId == 1) {
        std::cout << "SUCCESS: Detection parsed correctly." << std::endl;
    } else {
        std::cout << "FAIL: Incorrect parsing." << std::endl;
    }

    return 0;
}
