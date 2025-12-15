#include <iostream>

int main() {
#ifdef DEBUG_BUILD
    std::cout << "Debug Mode" << std::endl;
#else
    std::cout << "Release Mode" << std::endl;
#endif
    return 0;
}
