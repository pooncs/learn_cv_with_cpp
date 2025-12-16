#include <gtest/gtest.h>
#include <vector>
#include <opencv2/opencv.hpp>

struct Detection {
    cv::Rect box;
    float conf;
    int classId;
};

// Duplicate postprocess for test scope
std::vector<Detection> postprocess(float* data, int num_anchors, int num_classes, float conf_threshold) {
    std::vector<Detection> detections;
    for (int i = 0; i < num_anchors; ++i) {
        float cx = data[0 * num_anchors + i];
        float cy = data[1 * num_anchors + i];
        float w  = data[2 * num_anchors + i];
        float h  = data[3 * num_anchors + i];

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

TEST(PostProcessing, SingleDetection) {
    const int num_anchors = 1;
    const int num_classes = 1;
    // cx, cy, w, h, score
    float data[] = {100, 100, 50, 50, 0.8}; 
    
    auto res = postprocess(data, num_anchors, num_classes, 0.5f);
    
    EXPECT_EQ(res.size(), 1);
    EXPECT_EQ(res[0].box.x, 75); // 100 - 50/2
    EXPECT_EQ(res[0].box.y, 75);
    EXPECT_NEAR(res[0].conf, 0.8f, 1e-5);
}

TEST(PostProcessing, Threshold) {
    const int num_anchors = 1;
    const int num_classes = 1;
    float data[] = {100, 100, 50, 50, 0.4}; 
    
    auto res = postprocess(data, num_anchors, num_classes, 0.5f);
    EXPECT_EQ(res.size(), 0);
}
