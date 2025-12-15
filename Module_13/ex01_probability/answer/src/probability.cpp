#include "probability.h"
#include <iostream>

std::vector<double> convolve1D(const std::vector<double>& belief, const std::vector<double>& kernel) {
    int N = belief.size();
    int K = kernel.size();
    int offset = K / 2; // Assuming odd kernel size, center is at K/2
    std::vector<double> result(N, 0.0);
    
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < K; ++k) {
            // Motion model: x_t = x_{t-1} + u. 
            // If kernel represents probability of step sizes, centered at 'offset'.
            // If kernel[k] is prob of moving (k - offset) steps.
            // Then to end up at 'i', we must have come from 'i - (k - offset)'.
            int shift = k - offset;
            int prev_idx = i - shift;
            
            // Cyclic boundary condition
            prev_idx = (prev_idx % N + N) % N; 
            
            result[i] += belief[prev_idx] * kernel[k];
        }
    }
    return result;
}
