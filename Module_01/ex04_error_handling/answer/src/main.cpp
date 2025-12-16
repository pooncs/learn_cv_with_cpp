#include <iostream>
#include <cmath>
#include <optional>

// Modern function returning std::optional.
// Explicitly tells the caller: "I might not return a double".
std::optional<double> safe_sqrt(double x) {
    if (x < 0) {
        return std::nullopt; // Represents "Nothing"
    }
    return std::sqrt(x); // Implicit conversion to optional<double>
}

int main() {
    double input = -5.0;
    
    // ---------------------------------------------------------
    // Task 2: Handle the Result
    // ---------------------------------------------------------
    auto result = safe_sqrt(input);

    // 'if (result)' is equivalent to 'if (result.has_value())'
    if (result) {
        std::cout << "Result: " << *result << "\n"; // Dereference to get value
    } else {
        std::cout << "Calculation failed (Modern Check)\n";
    }

    // ---------------------------------------------------------
    // Task 3: Default Value
    // ---------------------------------------------------------
    // Returns the value if present, otherwise returns 0.0 (or whatever fallback you provide).
    // This is great for one-liners.
    std::cout << "Value or default: " << result.value_or(0.0) << "\n";
    
    // Test with valid input
    auto result2 = safe_sqrt(16.0);
    std::cout << "Valid input (16.0): " << result2.value_or(0.0) << "\n";

    return 0;
}
