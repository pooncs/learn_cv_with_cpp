#include <iostream>
#include <vector>
#include <map>
#include <string>

// Legacy function with verbose types
std::map<std::string, std::vector<int>> get_data() {
    return {{"ids", {1, 2, 3}}, {"scores", {10, 20, 30}}};
}

int main() {
    // Task 1: Refactor using auto
    // TODO: Replace the explicit type with 'auto'
    std::map<std::string, std::vector<int>> data = get_data();

    std::cout << "Iterating using auto:\n";

    // Task 2: Use auto in loop
    // TODO: Convert this legacy loop to a range-based for loop using 'const auto&'
    // for (const auto& pair : data) { ... }
    for (std::map<std::string, std::vector<int>>::iterator it = data.begin(); it != data.end(); ++it) {
        std::cout << it->first << ": ";
        for (size_t i = 0; i < it->second.size(); ++i) {
            std::cout << it->second[i] << " ";
        }
        std::cout << "\n";
    }

    // Task 3: decltype check
    // TODO: Use decltype to create 'copy_data' with the same type as 'data'
    // decltype(data) copy_data = ...;
    
    return 0;
}
