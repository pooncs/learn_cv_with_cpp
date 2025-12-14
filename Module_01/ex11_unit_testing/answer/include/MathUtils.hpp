#pragma once
#include <stdexcept>

class MathUtils {
public:
    static int factorial(int n) {
        if (n < 0) throw std::invalid_argument("Negative input");
        if (n == 0 || n == 1) return 1;
        return n * factorial(n - 1);
    }

    static int fibonacci(int n) {
        if (n < 0) throw std::invalid_argument("Negative input");
        if (n == 0) return 0;
        if (n == 1) return 1;
        return fibonacci(n - 1) + fibonacci(n - 2);
    }

    static double lerp(double a, double b, double t) {
        return a + t * (b - a);
    }
};
