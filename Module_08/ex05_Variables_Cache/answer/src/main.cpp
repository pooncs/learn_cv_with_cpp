#include <iostream>

int main() {
#ifdef MY_FEATURE
    std::cout << "Feature is ENABLED." << std::endl;
#else
    std::cout << "Feature is DISABLED." << std::endl;
#endif
    return 0;
}
