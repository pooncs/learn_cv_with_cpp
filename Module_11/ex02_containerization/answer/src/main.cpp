#include <iostream>
#include <vector>
#include <string>

int main() {
    std::cout << "==================================" << std::endl;
    std::cout << "   Hello from a Docker Container! " << std::endl;
    std::cout << "==================================" << std::endl;
    
    std::cout << "Running standard C++ code..." << std::endl;
    std::vector<std::string> msg = {"Containerization", "is", "cool"};
    for (const auto& w : msg) {
        std::cout << w << " ";
    }
    std::cout << std::endl;

    return 0;
}
