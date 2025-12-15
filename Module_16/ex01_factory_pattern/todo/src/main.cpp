#include "factory.hpp"
#include <iostream>

class MockAlgo : public IAlgorithm {
public:
    void process() override { std::cout << "Mock processing\n"; }
    std::string name() const override { return "Mock"; }
};

int main() {
    auto& factory = AlgorithmFactory::instance();
    
    // TODO: Register MockAlgo
    
    // TODO: Create MockAlgo
    
    std::cout << "Factory Pattern Todo" << std::endl;
    return 0;
}
