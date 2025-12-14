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
    // TODO: Use a lambda in std::sort
    // std::sort(...);

    std::cout << "Sorted:\n";
    for(const auto& d : dets) std::cout << d.class_name << ": " << d.confidence << "\n";

    // 2. Filter using std::copy_if
    std::vector<Detection> high_conf;
    float threshold = 0.6f;
    // TODO: Use std::copy_if with a capturing lambda

    return 0;
}
