#pragma once
#include <Eigen/Dense>
#include <vector>
#include <iostream>

struct DataPoint {
    double x;
    double y;
};

class CurveFitter {
public:
    // Params: [a, b, c] for ax^2 + bx + c
    using Vector3d = Eigen::Vector3d;

    static Vector3d solve(const std::vector<DataPoint>& data, int max_iters = 10) {
        Vector3d theta = Vector3d::Zero(); // Initial guess

        for (int iter = 0; iter < max_iters; ++iter) {
            Eigen::MatrixXd J(data.size(), 3);
            Eigen::VectorXd r(data.size());

            for (size_t i = 0; i < data.size(); ++i) {
                double x = data[i].x;
                double y_obs = data[i].y;
                double y_est = theta[0] * x * x + theta[1] * x + theta[2];

                r[i] = y_est - y_obs;

                // Jacobian
                J(i, 0) = x * x; // d/da
                J(i, 1) = x;     // d/db
                J(i, 2) = 1.0;   // d/dc
            }

            // H = J^T * J
            Eigen::Matrix3d H = J.transpose() * J;
            // b = -J^T * r
            Eigen::Vector3d b = -J.transpose() * r;

            // Solve H * delta = b
            Eigen::Vector3d delta = H.ldlt().solve(b);

            if (delta.norm() < 1e-6) break;

            theta += delta;
        }
        return theta;
    }
};
