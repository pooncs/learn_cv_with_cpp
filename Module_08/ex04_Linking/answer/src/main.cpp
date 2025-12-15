#include <iostream>
#include "lib_b.hpp"
// #include "lib_a.hpp" // Should fail if uncommented (unless paths coincide)

int main() {
    std::cout << "Val B: " << getValB() << std::endl;
    return 0;
}
