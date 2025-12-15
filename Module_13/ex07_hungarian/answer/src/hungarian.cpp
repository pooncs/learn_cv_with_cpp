#include "hungarian.h"
#include <algorithm>
#include <limits>
#include <cmath>

HungarianAlgorithm::HungarianAlgorithm() {}
HungarianAlgorithm::~HungarianAlgorithm() {}

// A standard implementation of the Hungarian Algorithm (Munkres)
double HungarianAlgorithm::Solve(const std::vector<std::vector<double>>& distMatrix, std::vector<int>& assignment) {
    int nRows = distMatrix.size();
    if (nRows == 0) return 0.0;
    int nCols = distMatrix[0].size();

    double cost = 0.0;

    // Fill distMatrixIn form standard vector
    std::vector<double> distMatrixIn(nRows * nCols);
    for (int i = 0; i < nRows; i++)
        for (int j = 0; j < nCols; j++)
            distMatrixIn[i + nRows * j] = distMatrix[i][j];

    // Assignment vector
    assignment.clear();
    assignment.resize(nRows, -1);

    assignmentOptimal(assignment, cost, distMatrixIn, nRows, nCols);

    return cost;
}

void HungarianAlgorithm::assignmentOptimal(std::vector<int>& assignment, double& cost, const std::vector<double>& distMatrixIn, int nOfRows, int nOfColumns) {
    double* distMatrix, * distMatrixTemp, * distMatrixEnd, * columnEnd, value, minValue;
    bool* coveredColumns, * coveredRows, * starMatrix, * newStarMatrix, * primeMatrix;
    int nOfElements, minDim, row, col;

    // Init
    cost = 0;
    for (row = 0; row < nOfRows; row++)
        assignment[row] = -1;

    nOfElements = nOfRows * nOfColumns;
    distMatrix = new double[nOfElements];
    distMatrixEnd = distMatrix + nOfElements;

    for (row = 0; row < nOfElements; row++)
        distMatrix[row] = distMatrixIn[row];

    coveredColumns = new bool[nOfColumns];
    coveredRows = new bool[nOfRows];
    starMatrix = new bool[nOfElements];
    primeMatrix = new bool[nOfElements];
    newStarMatrix = new bool[nOfElements];

    for (row = 0; row < nOfElements; row++) {
        starMatrix[row] = false;
        primeMatrix[row] = false;
        newStarMatrix[row] = false;
    }
    for (row = 0; row < nOfRows; row++) coveredRows[row] = false;
    for (col = 0; col < nOfColumns; col++) coveredColumns[col] = false;

    // Preliminary steps
    if (nOfRows <= nOfColumns) {
        minDim = nOfRows;
        for (row = 0; row < nOfRows; row++) {
            distMatrixTemp = distMatrix + row;
            minValue = *distMatrixTemp;
            distMatrixTemp += nOfRows;
            while (distMatrixTemp < distMatrixEnd) {
                value = *distMatrixTemp;
                if (value < minValue) minValue = value;
                distMatrixTemp += nOfRows;
            }
            distMatrixTemp = distMatrix + row;
            while (distMatrixTemp < distMatrixEnd) {
                *distMatrixTemp -= minValue;
                distMatrixTemp += nOfRows;
            }
        }
        for (row = 0; row < nOfRows; row++) {
            for (col = 0; col < nOfColumns; col++) {
                if (fabs(distMatrix[row + nOfRows * col]) < 1e-9) {
                    if (!coveredColumns[col]) {
                        starMatrix[row + nOfRows * col] = true;
                        coveredColumns[col] = true;
                        break;
                    }
                }
            }
        }
    } else {
        minDim = nOfColumns;
        for (col = 0; col < nOfColumns; col++) {
            distMatrixTemp = distMatrix + nOfRows * col;
            columnEnd = distMatrixTemp + nOfRows;
            minValue = *distMatrixTemp++;
            while (distMatrixTemp < columnEnd) {
                value = *distMatrixTemp++;
                if (value < minValue) minValue = value;
            }
            distMatrixTemp = distMatrix + nOfRows * col;
            while (distMatrixTemp < columnEnd) {
                *distMatrixTemp++ -= minValue;
            }
        }
        for (col = 0; col < nOfColumns; col++) {
            for (row = 0; row < nOfRows; row++) {
                if (fabs(distMatrix[row + nOfRows * col]) < 1e-9) {
                    if (!coveredRows[row]) {
                        starMatrix[row + nOfRows * col] = true;
                        coveredColumns[col] = true;
                        coveredRows[row] = true;
                        break;
                    }
                }
            }
        }
        for (row = 0; row < nOfRows; row++) coveredRows[row] = false;
    }

    step2b(assignment.data(), distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);

    computeassignmentcost(assignment, cost, distMatrixIn, nOfRows);

    delete[] distMatrix;
    delete[] coveredColumns;
    delete[] coveredRows;
    delete[] starMatrix;
    delete[] primeMatrix;
    delete[] newStarMatrix;
}

void HungarianAlgorithm::buildassignmentvector(std::vector<int>& assignment, bool* starMatrix, int nOfRows, int nOfColumns) {
    for (int row = 0; row < nOfRows; row++)
        for (int col = 0; col < nOfColumns; col++)
            if (starMatrix[row + nOfRows * col]) {
                assignment[row] = col;
                break;
            }
}

void HungarianAlgorithm::computeassignmentcost(const std::vector<int>& assignment, double& cost, const std::vector<double>& distMatrix, int nOfRows) {
    for (int row = 0; row < nOfRows; row++) {
        int col = assignment[row];
        if (col >= 0)
            cost += distMatrix[row + nOfRows * col];
    }
}

void HungarianAlgorithm::step2a(int* assignment, double* distMatrix, bool* starMatrix, bool* newStarMatrix, bool* primeMatrix, bool* coveredColumns, bool* coveredRows, int nOfRows, int nOfColumns, int minDim) {
    bool* starMatrixTemp, * columnEnd;
    int col;
    for (col = 0; col < nOfColumns; col++) {
        starMatrixTemp = starMatrix + nOfRows * col;
        columnEnd = starMatrixTemp + nOfRows;
        while (starMatrixTemp < columnEnd) {
            if (*starMatrixTemp++) {
                coveredColumns[col] = true;
                break;
            }
        }
    }
    step2b(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

void HungarianAlgorithm::step2b(int* assignment, double* distMatrix, bool* starMatrix, bool* newStarMatrix, bool* primeMatrix, bool* coveredColumns, bool* coveredRows, int nOfRows, int nOfColumns, int minDim) {
    int col, nOfCoveredColumns = 0;
    for (col = 0; col < nOfColumns; col++)
        if (coveredColumns[col]) nOfCoveredColumns++;

    if (nOfCoveredColumns == minDim) {
        buildassignmentvector(*(std::vector<int>*) & assignment, starMatrix, nOfRows, nOfColumns); // Cast hack, in real class vector is member or passed by ref
        // Actually we passed vector<int>& but using .data() converted to int*. 
        // We need to re-populate the vector.
        std::vector<int> temp(nOfRows);
        buildassignmentvector(temp, starMatrix, nOfRows, nOfColumns);
        for(int i=0; i<nOfRows; ++i) assignment[i] = temp[i];
    } else {
        step3(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
    }
}

void HungarianAlgorithm::step3(int* assignment, double* distMatrix, bool* starMatrix, bool* newStarMatrix, bool* primeMatrix, bool* coveredColumns, bool* coveredRows, int nOfRows, int nOfColumns, int minDim) {
    bool zerosFound = true;
    while (zerosFound) {
        zerosFound = false;
        for (int col = 0; col < nOfColumns; col++) {
            if (!coveredColumns[col]) {
                for (int row = 0; row < nOfRows; row++) {
                    if ((!coveredRows[row]) && (fabs(distMatrix[row + nOfRows * col]) < 1e-9)) {
                        primeMatrix[row + nOfRows * col] = true;
                        int starCol = -1;
                        for (int c = 0; c < nOfColumns; c++)
                            if (starMatrix[row + nOfRows * c]) {
                                starCol = c;
                                break;
                            }
                        if (starCol >= 0) {
                            coveredRows[row] = true;
                            coveredColumns[starCol] = false;
                            zerosFound = true;
                            break;
                        } else {
                            step4(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim, row, col);
                            return;
                        }
                    }
                }
            }
        }
    }
    step5(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

void HungarianAlgorithm::step4(int* assignment, double* distMatrix, bool* starMatrix, bool* newStarMatrix, bool* primeMatrix, bool* coveredColumns, bool* coveredRows, int nOfRows, int nOfColumns, int minDim, int row, int col) {
    int n, starRow, starCol, primeRow, primeCol;
    int nOfElements = nOfRows * nOfColumns;

    for (n = 0; n < nOfElements; n++) newStarMatrix[n] = starMatrix[n];
    newStarMatrix[row + nOfRows * col] = true;

    starCol = col;
    while (true) {
        starRow = -1;
        for (n = 0; n < nOfRows; n++)
            if (starMatrix[n + nOfRows * starCol]) {
                starRow = n;
                break;
            }
        if (starRow < 0) break;
        newStarMatrix[starRow + nOfRows * starCol] = false;
        primeRow = starRow;
        for (n = 0; n < nOfColumns; n++)
            if (primeMatrix[primeRow + nOfRows * n]) {
                primeCol = n;
                break;
            }
        newStarMatrix[primeRow + nOfRows * primeCol] = true;
        starCol = primeCol;
    }
    for (n = 0; n < nOfElements; n++) {
        primeMatrix[n] = false;
        starMatrix[n] = newStarMatrix[n];
    }
    for (n = 0; n < nOfRows; n++) coveredRows[n] = false;
    for (n = 0; n < nOfColumns; n++) coveredColumns[n] = false;

    step2a(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

void HungarianAlgorithm::step5(int* assignment, double* distMatrix, bool* starMatrix, bool* newStarMatrix, bool* primeMatrix, bool* coveredColumns, bool* coveredRows, int nOfRows, int nOfColumns, int minDim) {
    double h, value;
    int row, col;

    h = std::numeric_limits<double>::max();
    for (row = 0; row < nOfRows; row++)
        if (!coveredRows[row])
            for (col = 0; col < nOfColumns; col++)
                if (!coveredColumns[col]) {
                    value = distMatrix[row + nOfRows * col];
                    if (value < h) h = value;
                }

    for (row = 0; row < nOfRows; row++)
        if (coveredRows[row])
            for (col = 0; col < nOfColumns; col++)
                distMatrix[row + nOfRows * col] += h;

    for (col = 0; col < nOfColumns; col++)
        if (!coveredColumns[col])
            for (row = 0; row < nOfRows; row++)
                distMatrix[row + nOfRows * col] -= h;

    step3(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}
