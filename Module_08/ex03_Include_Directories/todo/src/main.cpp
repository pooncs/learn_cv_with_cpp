#include <iostream>
#include "math_utils.hpp" // This should work if include directories are set correctly

int main() {
    std::cout << "3 + 4 = " << math::add(3, 4) << std::endl;
    return 0;
}
