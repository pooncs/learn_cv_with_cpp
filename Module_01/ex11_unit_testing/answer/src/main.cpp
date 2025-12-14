#include <iostream>
#include "MathUtils.hpp"

int main() {
    std::cout << "Factorial(5): " << MathUtils::factorial(5) << "\n";
    std::cout << "Fibonacci(6): " << MathUtils::fibonacci(6) << "\n";
    std::cout << "Lerp(0, 10, 0.5): " << MathUtils::lerp(0, 10, 0.5) << "\n";
    return 0;
}
