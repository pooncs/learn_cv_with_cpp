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
    auto data = get_data();

    std::cout << "Iterating using auto:\n";
    // TODO: Use auto in loop
    for (const auto& pair : data) {
        std::cout << pair.first << ": ";
        for (auto val : pair.second) {
            std::cout << val << " ";
        }
        std::cout << "\n";
    }

    // TODO: decltype check
    decltype(data) copy_data = data;
    std::cout << "Copy size: " << copy_data.size() << "\n";

    return 0;
}
