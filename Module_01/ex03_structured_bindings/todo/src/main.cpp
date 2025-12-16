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
    
    // Legacy way
    double old_x = p.x;
    double old_y = p.y;
    double old_z = p.z;
    
    // TODO: Use structured binding to unpack 'p' into x, y, z
    // auto [x, y, z] = ...

    std::cout << "Point: " << old_x << ", " << old_y << ", " << old_z << "\n";


    // ---------------------------------------------------------
    // Task 2: Unpacking a Tuple
    // ---------------------------------------------------------
    // TODO: Call get_config() and bind to id, threshold, filename
    // auto [id, th, name] = ...
    
    // Placeholder to make it compile
    auto config = get_config();
    std::cout << "Config ID: " << std::get<0>(config) << "\n";


    // ---------------------------------------------------------
    // Task 3: Iterating a Map
    // ---------------------------------------------------------
    std::map<std::string, int> scores = {{"Alice", 90}, {"Bob", 85}, {"Charlie", 95}};

    std::cout << "Scores:\n";
    // TODO: Refactor loop to use structured binding
    for (std::map<std::string, int>::iterator it = scores.begin(); it != scores.end(); ++it) {
        std::cout << it->first << ": " << it->second << "\n";
    }

    return 0;
}
