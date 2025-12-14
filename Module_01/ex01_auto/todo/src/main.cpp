#include <iostream>
#include <vector>
#include <map>
#include <string>

// Legacy function with verbose types
std::map<std::string, std::vector<int>> get_data() {
    return {{"ids", {1, 2, 3}}, {"scores", {10, 20, 30}}};
}

int main() {
    // TODO: Refactor using auto
    // std::map<std::string, std::vector<int>> data = get_data();
    auto data = get_data(); // Placeholder, but try to understand why

    std::cout << "Iterating using auto:\n";
    // TODO: Use auto in loop
    // for (std::map<std::string, std::vector<int>>::iterator it = ...)
    
    // TODO: decltype check

    return 0;
}
