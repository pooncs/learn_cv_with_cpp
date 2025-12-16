#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <algorithm>

struct Detection {
    cv::Rect box;
    float conf;
    int classId;
};

// 1. IoU Function
float iou(const cv::Rect& a, const cv::Rect& b) {
    cv::Rect intersection = a & b; // Intersection
    float interArea = intersection.area();
    float unionArea = a.area() + b.area() - interArea;
    
    if (unionArea <= 0.0f) return 0.0f;
    return interArea / unionArea;
}

// 2. NMS Implementation
std::vector<Detection> nms(std::vector<Detection>& dets, float threshold) {
    if (dets.empty()) return {};

    // Sort by confidence descending
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

            // Only suppress if same class (optional, depending on requirements)
            if (dets[i].classId == dets[j].classId) {
                if (iou(dets[i].box, dets[j].box) > threshold) {
                    suppressed[j] = true;
                }
            }
        }
    }
    return kept;
}

int main() {
    std::vector<Detection> dets;
    // Box 1: Strong confidence
    dets.push_back({cv::Rect(100, 100, 50, 50), 0.9f, 0});
    // Box 2: Overlapping Box 1, lower confidence -> Should be suppressed
    dets.push_back({cv::Rect(105, 105, 50, 50), 0.8f, 0});
    // Box 3: Disjoint, lower confidence -> Should be kept
    dets.push_back({cv::Rect(300, 300, 50, 50), 0.7f, 0});

    std::cout << "Original count: " << dets.size() << std::endl;

    auto result = nms(dets, 0.5f);

    std::cout << "After NMS count: " << result.size() << std::endl;
    for (const auto& d : result) {
        std::cout << "Box: " << d.box << " Conf: " << d.conf << std::endl;
    }

    return 0;
}
