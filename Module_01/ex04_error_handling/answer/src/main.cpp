#include <iostream>
#include <optional>
#include <cmath>

std::optional<double> safe_sqrt(double x) {
    if (x < 0) return std::nullopt;
    return std::sqrt(x);
}

int main() {
    double val = -1.0;
    
    // Using has_value()
    auto res = safe_sqrt(val);
    if (res.has_value()) {
        std::cout << "Sqrt: " << res.value() << "\n";
    } else {
        std::cout << "Error: Negative input\n";
    }

    // Using value_or()
    std::cout << "Safe value: " << safe_sqrt(val).value_or(0.0) << "\n";

    return 0;
}
