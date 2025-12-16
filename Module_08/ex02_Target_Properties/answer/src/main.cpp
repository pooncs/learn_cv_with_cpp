#include <iostream>
#include <string_view>

int main() {
    // C++17 feature: string_view
    std::string_view sv = "Hello, C++17!";
    std::cout << sv << std::endl;

#ifdef MY_FEATURE
    std::cout << "MY_FEATURE is enabled!" << std::endl;
#else
    std::cout << "MY_FEATURE is disabled." << std::endl;
#endif

    return 0;
}
