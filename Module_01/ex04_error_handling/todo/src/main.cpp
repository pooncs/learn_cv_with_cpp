#include <iostream>
#include <cmath>
#include <optional> // You will need this

// Legacy function that returns -1.0 on failure.
// This is bad because -1.0 could theoretically be a result in other contexts,
// and the user might forget to check for it.
double safe_sqrt(double x) {
    if (x < 0) {
        return -1.0; // Error code
    }
    return std::sqrt(x);
}

int main() {
    // ---------------------------------------------------------
    // Task 1: Refactor safe_sqrt
    // ---------------------------------------------------------
    // TODO: Change safe_sqrt signature to return std::optional<double>
    // If x < 0, return std::nullopt;
    // Otherwise, return result;

    double input = -5.0;
    
    // ---------------------------------------------------------
    // Task 2: Handle the Result
    // ---------------------------------------------------------
    // TODO: Update the call site.
    // auto result = safe_sqrt(input);
    double result = safe_sqrt(input);

    if (result == -1.0) {
         std::cout << "Calculation failed (Legacy Check)\n";
    } else {
         std::cout << "Result: " << result << "\n";
    }

    // TODO: Use if(result.has_value()) ...


    // ---------------------------------------------------------
    // Task 3: Default Value
    // ---------------------------------------------------------
    // TODO: Use .value_or() to print 0.0 if failed
    // std::cout << "Value or default: " << ... << "\n";

    return 0;
}
