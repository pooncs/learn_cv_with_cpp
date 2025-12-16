#include <iostream>
#include <vector>
#include "nms.h"

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
