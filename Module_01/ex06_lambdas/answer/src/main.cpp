#include <iostream>
#include <vector>
#include <algorithm>
#include <string>

struct Detection {
    int id;
    float confidence;
    std::string class_name;
};

int main() {
    std::vector<Detection> dets = {
        {1, 0.9f, "car"},
        {2, 0.5f, "person"},
        {3, 0.95f, "car"},
        {4, 0.2f, "tree"}
    };

    // 1. Sort by confidence (descending)
    std::sort(dets.begin(), dets.end(), [](const Detection& a, const Detection& b) {
        return a.confidence > b.confidence;
    });

    std::cout << "Sorted:\n";
    for(const auto& d : dets) std::cout << d.class_name << ": " << d.confidence << "\n";

    // 2. Filter using std::copy_if
    std::vector<Detection> high_conf;
    float threshold = 0.6f;
    std::copy_if(dets.begin(), dets.end(), std::back_inserter(high_conf), [threshold](const Detection& d) {
        return d.confidence > threshold;
    });

    std::cout << "\nFiltered (> " << threshold << "):\n";
    for(const auto& d : high_conf) std::cout << d.class_name << ": " << d.confidence << "\n";

    return 0;
}
