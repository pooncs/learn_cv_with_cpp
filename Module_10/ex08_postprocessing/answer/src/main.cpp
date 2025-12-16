#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <algorithm>

struct Detection {
    cv::Rect box;
    float conf;
    int classId;
};

// Data layout assumption: [Batch=1, Channels=(4 coords + classes), Anchors]
// But typical YOLOv8 export might be [1, 84, 8400].
// We iterate over 8400 anchors. For each, we check 80 classes.
std::vector<Detection> postprocess(float* data, int num_anchors, int num_classes, float conf_threshold) {
    std::vector<Detection> detections;
    int dimensions = 4 + num_classes; // e.g. 4 + 80 = 84

    // Iterate over anchors
    for (int i = 0; i < num_anchors; ++i) {
        // Data is column-major or row-major?
        // Usually YOLOv8 output is [1, 84, 8400].
        // To access anchor 'i':
        // cx = data[0 * num_anchors + i]
        // cy = data[1 * num_anchors + i]
        // w  = data[2 * num_anchors + i]
        // h  = data[3 * num_anchors + i]
        // class scores = data[4... * num_anchors + i]

        float cx = data[0 * num_anchors + i];
        float cy = data[1 * num_anchors + i];
        float w  = data[2 * num_anchors + i];
        float h  = data[3 * num_anchors + i];

        // Find max class score
        float max_score = -1.0f;
        int max_class_id = -1;

        for (int c = 0; c < num_classes; ++c) {
            float score = data[(4 + c) * num_anchors + i];
            if (score > max_score) {
                max_score = score;
                max_class_id = c;
            }
        }

        if (max_score > conf_threshold) {
            int left = static_cast<int>(cx - w / 2);
            int top = static_cast<int>(cy - h / 2);
            int width = static_cast<int>(w);
            int height = static_cast<int>(h);

            detections.push_back({cv::Rect(left, top, width, height), max_score, max_class_id});
        }
    }

    return detections;
}

int main() {
    const int num_anchors = 10;
    const int num_classes = 2; // e.g., Person, Car
    const int dims = 4 + num_classes;
    
    // Create dummy data: [1, 6, 10] flat array
    // Layout: 6 rows (cx, cy, w, h, c0, c1), 10 columns (anchors)
    std::vector<float> data(dims * num_anchors, 0.0f);

    // Make anchor 5 a detection
    // cx=100, cy=100, w=50, h=50
    // class 1 score = 0.9
    int idx = 5;
    data[0 * num_anchors + idx] = 100.0f;
    data[1 * num_anchors + idx] = 100.0f;
    data[2 * num_anchors + idx] = 50.0f;
    data[3 * num_anchors + idx] = 50.0f;
    data[4 * num_anchors + idx] = 0.1f; // Class 0
    data[5 * num_anchors + idx] = 0.9f; // Class 1

    auto dets = postprocess(data.data(), num_anchors, num_classes, 0.5f);

    std::cout << "Detections found: " << dets.size() << std::endl;
    for (const auto& det : dets) {
        std::cout << "Class: " << det.classId 
                  << " Conf: " << det.conf 
                  << " Box: " << det.box << std::endl;
    }

    return 0;
}
