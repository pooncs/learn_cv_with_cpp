#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "nms.h"

int test_iou() {
    cv::Rect a(0, 0, 10, 10);
    cv::Rect b(0, 0, 10, 10);
    // Intersection = 100, Union = 100, IoU = 1.0
    if (std::abs(iou(a, b) - 1.0f) > 1e-5) {
        std::cerr << "IoU Test 1 Failed" << std::endl;
        return 1;
    }

    cv::Rect c(10, 0, 10, 10);
    // Intersection = 0, Union = 200, IoU = 0.0
    if (std::abs(iou(a, c) - 0.0f) > 1e-5) {
        std::cerr << "IoU Test 2 Failed" << std::endl;
        return 1;
    }

    cv::Rect d(5, 0, 10, 10);
    // Intersection: x=[5,10], y=[0,10] -> w=5, h=10 -> area=50
    // Union: 100 + 100 - 50 = 150
    // IoU: 50/150 = 0.3333
    if (std::abs(iou(a, d) - 0.333333f) > 1e-4) {
        std::cerr << "IoU Test 3 Failed: " << iou(a, d) << std::endl;
        return 1;
    }

    return 0;
}

int test_nms() {
    std::vector<Detection> dets;
    // Box 1: Strong confidence
    dets.push_back({cv::Rect(100, 100, 50, 50), 0.9f, 0});
    // Box 2: Overlapping Box 1 (high overlap), lower confidence -> Should be suppressed
    dets.push_back({cv::Rect(102, 102, 50, 50), 0.8f, 0});
    // Box 3: Disjoint, lower confidence -> Should be kept
    dets.push_back({cv::Rect(300, 300, 50, 50), 0.7f, 0});

    auto result = nms(dets, 0.5f);

    if (result.size() != 2) {
        std::cerr << "NMS Test Failed: Expected 2 boxes, got " << result.size() << std::endl;
        return 1;
    }

    if (result[0].conf != 0.9f) {
         std::cerr << "NMS Test Failed: First box should be 0.9 confidence" << std::endl;
         return 1;
    }

    if (result[1].conf != 0.7f) {
         std::cerr << "NMS Test Failed: Second box should be 0.7 confidence" << std::endl;
         return 1;
    }

    return 0;
}

int main(int argc, char** argv) {
    if (argc > 1) {
        std::string test_name = argv[1];
        if (test_name == "iou") return test_iou();
        if (test_name == "nms") return test_nms();
    }

    // Run all if no arg
    if (test_iou() != 0) return 1;
    if (test_nms() != 0) return 1;
    
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
