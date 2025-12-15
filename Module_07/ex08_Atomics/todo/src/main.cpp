#include <iostream>
#include <thread>
#include <vector>
#include "atomic_counter.hpp"

int main() {
    cv_curriculum::AtomicCounter counter;
    // Simple test
    counter.increment();
    if (counter.get() == 1) std::cout << "Incremented." << std::endl;
    else std::cout << "Failed." << std::endl;
    return 0;
}
