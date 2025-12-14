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
    auto [x, y, z] = p;
    std::cout << "Point: " << x << ", " << y << ", " << z << "\n";

    // 2. Tuple binding
    auto [id, threshold, filename] = get_config();
    std::cout << "Config: " << id << ", " << threshold << ", " << filename << "\n";

    // 3. Map binding
    std::map<std::string, int> scores = {{"Alice", 90}, {"Bob", 85}};
    for (const auto& [name, score] : scores) {
        std::cout << name << ": " << score << "\n";
    }

    return 0;
}
