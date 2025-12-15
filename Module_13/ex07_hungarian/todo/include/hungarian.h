#pragma once
#include <vector>

class HungarianAlgorithm {
public:
    HungarianAlgorithm();
    ~HungarianAlgorithm();

    double Solve(const std::vector<std::vector<double>>& distMatrix, std::vector<int>& assignment);

private:
    // TODO: Helper methods for Hungarian steps
};
