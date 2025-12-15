#pragma once
#include <vector>
#include <iostream>
#include <limits>
#include <cmath>

class HungarianAlgorithm {
public:
    HungarianAlgorithm();
    ~HungarianAlgorithm();

    double Solve(const std::vector<std::vector<double>>& distMatrix, std::vector<int>& assignment);

private:
    void assignmentOptimal(std::vector<int>& assignment, double& cost, const std::vector<double>& distMatrixIn, int nOfRows, int nOfColumns);
    void buildassignmentvector(std::vector<int>& assignment, bool* starMatrix, int nOfRows, int nOfColumns);
    void computeassignmentcost(const std::vector<int>& assignment, double& cost, const std::vector<double>& distMatrix, int nOfRows);
    void step2a(int* assignment, double* distMatrix, bool* starMatrix, bool* newStarMatrix, bool* primeMatrix, bool* coveredColumns, bool* coveredRows, int nOfRows, int nOfColumns, int minDim);
    void step2b(int* assignment, double* distMatrix, bool* starMatrix, bool* newStarMatrix, bool* primeMatrix, bool* coveredColumns, bool* coveredRows, int nOfRows, int nOfColumns, int minDim);
    void step3(int* assignment, double* distMatrix, bool* starMatrix, bool* newStarMatrix, bool* primeMatrix, bool* coveredColumns, bool* coveredRows, int nOfRows, int nOfColumns, int minDim);
    void step4(int* assignment, double* distMatrix, bool* starMatrix, bool* newStarMatrix, bool* primeMatrix, bool* coveredColumns, bool* coveredRows, int nOfRows, int nOfColumns, int minDim, int row, int col);
    void step5(int* assignment, double* distMatrix, bool* starMatrix, bool* newStarMatrix, bool* primeMatrix, bool* coveredColumns, bool* coveredRows, int nOfRows, int nOfColumns, int minDim);
};
