#include <iostream>
#include <optional>
#include <cmath>

// TODO: Change return type to std::optional<double>
double safe_sqrt(double x) {
    if (x < 0) return -1.0; // Error code
    return std::sqrt(x);
}

int main() {
    double val = -1.0;
    
    // TODO: Call safe_sqrt and handle the optional result using .has_value()
    
    // TODO: Use .value_or(0.0)

    return 0;
}
