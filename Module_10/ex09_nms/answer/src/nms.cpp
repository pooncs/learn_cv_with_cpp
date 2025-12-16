#include "nms.h"
#include <algorithm>

float iou(const cv::Rect& a, const cv::Rect& b) {
    cv::Rect intersection = a & b; // Intersection
    float interArea = intersection.area();
    float unionArea = a.area() + b.area() - interArea;
    
    if (unionArea <= 0.0f) return 0.0f;
    return interArea / unionArea;
}

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
