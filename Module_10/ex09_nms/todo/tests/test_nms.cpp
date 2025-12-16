#include <gtest/gtest.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <algorithm>

struct Detection {
    cv::Rect box;
    float conf;
    int classId;
};

float iou(const cv::Rect& a, const cv::Rect& b) {
    cv::Rect intersection = a & b; 
    float interArea = intersection.area();
    float unionArea = a.area() + b.area() - interArea;
    if (unionArea <= 0.0f) return 0.0f;
    return interArea / unionArea;
}

std::vector<Detection> nms(std::vector<Detection>& dets, float threshold) {
    if (dets.empty()) return {};
    std::sort(dets.begin(), dets.end(), [](const Detection& a, const Detection& b) {
        return a.conf > b.conf;
    });
    std::vector<Detection> kept;
    std::vector<bool> suppressed(dets.size(), false);
    for (size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) continue;
        kept.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (suppressed[j]) continue;
            if (dets[i].classId == dets[j].classId) {
                if (iou(dets[i].box, dets[j].box) > threshold) {
                    suppressed[j] = true;
                }
            }
        }
    }
    return kept;
}

TEST(NMS, Overlap) {
    std::vector<Detection> dets;
    dets.push_back({cv::Rect(0, 0, 10, 10), 0.9, 0});
    dets.push_back({cv::Rect(1, 1, 10, 10), 0.8, 0}); // High overlap
    dets.push_back({cv::Rect(100, 100, 10, 10), 0.7, 0}); // No overlap

    auto res = nms(dets, 0.5);
    EXPECT_EQ(res.size(), 2);
    EXPECT_NEAR(res[0].conf, 0.9, 1e-5);
    EXPECT_NEAR(res[1].conf, 0.7, 1e-5);
}
