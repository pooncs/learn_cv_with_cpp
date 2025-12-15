#pragma once
#include <Eigen/Dense>
#include <vector>

struct DataPoint {
    double x;
    double y;
};

class CurveFitter {
public:
    using Vector3d = Eigen::Vector3d;

    static Vector3d solve(const std::vector<DataPoint>& data, int max_iters = 10) {
        // TODO: Implement Gauss-Newton
        // 1. Initialize params
        // 2. Loop:
        //    a. Compute Jacobian (Nx3) and Residuals (Nx1)
        //    b. Solve normal equations: (J^T J) delta = -J^T r
        //    c. Update params
        return Vector3d::Zero();
    }
};
