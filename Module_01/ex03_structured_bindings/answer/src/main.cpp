#include <iostream>
#include <tuple>
#include <string>
#include <map>

struct Point {
    double x, y, z;
};

std::tuple<int, double, std::string> get_config() {
    return {101, 0.75, "config.json"};
}

int main() {
    // ---------------------------------------------------------
    // Task 1: Unpacking a Struct
    // ---------------------------------------------------------
    Point p{1.0, 2.0, 3.0};
    
    // Structured binding makes it easy to unpack public members.
    // 'auto' copies values. 'auto&' would bind by reference.
    auto [x, y, z] = p;

    std::cout << "Point: " << x << ", " << y << ", " << z << "\n";

    // ---------------------------------------------------------
    // Task 2: Unpacking a Tuple
    // ---------------------------------------------------------
    // We can unpack different types (int, double, string) directly.
    auto [id, threshold, filename] = get_config();
    
    std::cout << "Config: ID=" << id 
              << ", Th=" << threshold 
              << ", File=" << filename << "\n";

    // ---------------------------------------------------------
    // Task 3: Iterating a Map
    // ---------------------------------------------------------
    std::map<std::string, int> scores = {{"Alice", 90}, {"Bob", 85}, {"Charlie", 95}};

    std::cout << "Scores:\n";
    
    // 'const auto&' prevents copying the std::pair and the key/value inside it.
    // [name, score] unpacks pair.first and pair.second.
    for (const auto& [name, score] : scores) {
        std::cout << name << ": " << score << "\n";
    }

    return 0;
}
