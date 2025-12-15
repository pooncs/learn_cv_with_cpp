#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) return 1;
    std::string arg = argv[1];
    
    if (arg == "success") {
        std::cout << "Test passed!" << std::endl;
        return 0;
    } else {
        std::cout << "Test failed!" << std::endl;
        return 1;
    }
}
