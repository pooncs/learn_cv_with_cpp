#include <iostream>
#include "foo.h"

int main() {
    std::cout << "Found foo.h!" << std::endl;
    // We can't actually call foo() because foo.lib is fake/empty in this exercise
    // But if it linked successfully, CMake did its job finding the library file.
    return 0;
}
