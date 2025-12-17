#include "data_utils.hpp"
#include <iostream>
#include <map>
#include <string>
#include <vector>

int main() {
  // Task 1: Refactor using auto
  // Replaced std::map<std::string, std::vector<int>> with auto
  auto data = get_data();

  std::cout << "Iterating using auto:\n";

  // Task 2: Use auto in loop
  // Using const auto& to avoid copying the pair (std::pair<const std::string,
  // std::vector<int>>) Structured binding (C++17) could also be used: for
  // (const auto& [key, val] : data) but the prompt asked for range-based for
  // loop.
  for (const auto &pair : data) {
    std::cout << pair.first << ": ";
    for (const auto &val : pair.second) {
      std::cout << val << " ";
    }
    std::cout << "\n";
  }

  // Task 3: decltype check
  // Create 'copy_data' with the exact same type as 'data'
  decltype(data) copy_data = data;

  // Just to show it works
  std::cout << "Copy size: " << copy_data.size() << "\n";

  return 0;
}
