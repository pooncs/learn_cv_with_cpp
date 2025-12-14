#include <iostream>
#include <tuple>
#include <map>

struct Point {
    double x, y, z;
};

std::tuple<int, double, std::string> get_config() {
    return {1, 0.5, "config.json"};
}

int main() {
    // 1. Struct binding
    Point p = {1.0, 2.0, 3.0};
    // TODO: Unpack p into x, y, z using structured binding
    
    // 2. Tuple binding
    // TODO: Unpack get_config() result

    // 3. Map binding
    std::map<std::string, int> scores = {{"Alice", 90}, {"Bob", 85}};
    // TODO: Iterate over map using structured binding
    for (const auto& pair : scores) {
        // Refactor this line
    }

    return 0;
}
