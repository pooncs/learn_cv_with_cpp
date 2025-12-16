#include <iostream>
#include <vector>
#include <map>
#include <string>

// Legacy function with verbose types.
// Simulates a function that returns a complex data structure (e.g., detected object IDs and their scores).
std::map<std::string, std::vector<int>> get_data() {
    return {{"ids", {1, 2, 3}}, {"scores", {10, 20, 30}}};
}

int main() {
    // ---------------------------------------------------------
    // Task 1: Refactor using auto
    // ---------------------------------------------------------
    // OLD: std::map<std::string, std::vector<int>> data = get_data();
    // NEW: The compiler deduces the type of 'data' from the return type of get_data().
    auto data = get_data();

    std::cout << "Iterating using auto:\n";

    // ---------------------------------------------------------
    // Task 2: Use auto in loop
    // ---------------------------------------------------------
    // OLD: for(std::map<std::string, std::vector<int>>::iterator it = data.begin(); ...)
    // NEW: Range-based for loop with 'const auto&'.
    // We use 'const auto&' because:
    // 1. 'const': We don't intend to modify the data.
    // 2. '&': We want a reference to avoid copying the std::pair (which contains vectors!).
    for (const auto& pair : data) {
        std::cout << pair.first << ": ";
        
        // Inner loop: 'val' is an int.
        // Since int is small (primitive), 'auto val' (copy) is fine and efficient.
        // 'auto& val' would also work but isn't strictly necessary for ints.
        for (auto val : pair.second) {
            std::cout << val << " ";
        }
        std::cout << "\n";
    }

    // ---------------------------------------------------------
    // Task 3: decltype check
    // ---------------------------------------------------------
    // 'decltype(data)' gives us the exact type of the variable 'data',
    // which is std::map<std::string, std::vector<int>>.
    // This is useful in template metaprogramming or when the type is unnamed/unknown.
    decltype(data) copy_data = data; 
    
    std::cout << "Copy size: " << copy_data.size() << "\n";

    return 0;
}
