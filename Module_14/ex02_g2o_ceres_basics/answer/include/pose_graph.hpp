#pragma once
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <cmath>

struct Pose2D {
    double x, y, theta;

    // Composition operator: P_new = P_this * P_other
    Pose2D operator*(const Pose2D& other) const {
        double s = std::sin(theta);
        double c = std::cos(theta);
        return {
            x + c * other.x - s * other.y,
            y + s * other.x + c * other.y,
            theta + other.theta
        };
    }

    // Inverse
    Pose2D inverse() const {
        double s = std::sin(theta);
        double c = std::cos(theta);
        return {
            -c * x - s * y,
            s * x - c * y,
            -theta
        };
    }
};

struct Edge {
    int from_idx;
    int to_idx;
    Pose2D measurement; // Relative transform from->to
    Eigen::Matrix3d information; // Inverse covariance
};

class PoseGraphOptimizer {
public:
    static void optimize(std::vector<Pose2D>& nodes, const std::vector<Edge>& edges, int iterations = 10) {
        int n = nodes.size();
        for (int iter = 0; iter < iterations; ++iter) {
            // State vector size: 3 * n
            Eigen::VectorXd b = Eigen::VectorXd::Zero(3 * n);
            Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3 * n, 3 * n);

            double total_error = 0;

            for (const auto& edge : edges) {
                const Pose2D& xi = nodes[edge.from_idx];
                const Pose2D& xj = nodes[edge.to_idx];
                const Pose2D& z = edge.measurement;

                // Error function e = z^-1 * (xi^-1 * xj)
                // Simplified error for 2D:
                // dx = xj.x - xi.x
                // dy = xj.y - xi.y
                // e_x =  cos(xi.th)*dx + sin(xi.th)*dy - z.x
                // e_y = -sin(xi.th)*dx + cos(xi.th)*dy - z.y
                // e_th = xj.th - xi.th - z.th

                double dx = xj.x - xi.x;
                double dy = xj.y - xi.y;
                double c = std::cos(xi.theta);
                double s = std::sin(xi.theta);

                Eigen::Vector3d e;
                e(0) = c * dx + s * dy - z.x;
                e(1) = -s * dx + c * dy - z.y;
                e(2) = xj.theta - xi.theta - z.theta;
                
                // Normalize angle error
                while (e(2) > M_PI) e(2) -= 2 * M_PI;
                while (e(2) < -M_PI) e(2) += 2 * M_PI;

                total_error += e.transpose() * edge.information * e;

                // Jacobians (approximate/linearized)
                // A = d(e)/d(xi), B = d(e)/d(xj)
                Eigen::Matrix3d A, B;
                
                // Very simplified Jacobians for tutorial purposes
                // Real implementation involves derivatives of rotation matrices
                A.setZero();
                A(0, 0) = -c; A(0, 1) = -s; A(0, 2) = -s * dx + c * dy;
                A(1, 0) = s;  A(1, 1) = -c; A(1, 2) = -c * dx - s * dy;
                A(2, 0) = 0;  A(2, 1) = 0;  A(2, 2) = -1;

                B.setZero();
                B(0, 0) = c; B(0, 1) = s; B(0, 2) = 0;
                B(1, 0) = -s; B(1, 1) = c; B(1, 2) = 0;
                B(2, 0) = 0; B(2, 1) = 0; B(2, 2) = 1;

                // Fill H and b
                // H_ii += A^T * Omega * A
                int idx_i = 3 * edge.from_idx;
                int idx_j = 3 * edge.to_idx;

                H.block<3, 3>(idx_i, idx_i) += A.transpose() * edge.information * A;
                H.block<3, 3>(idx_i, idx_j) += A.transpose() * edge.information * B;
                H.block<3, 3>(idx_j, idx_i) += B.transpose() * edge.information * A;
                H.block<3, 3>(idx_j, idx_j) += B.transpose() * edge.information * B;

                b.segment<3>(idx_i) += A.transpose() * edge.information * e;
                b.segment<3>(idx_j) += B.transpose() * edge.information * e;
            }

            // Fix first node (gauge freedom)
            H.block<3, 3>(0, 0) += Eigen::Matrix3d::Identity() * 1e6;

            // Solve H * delta = -b
            Eigen::VectorXd delta = -H.ldlt().solve(b);

            // Update
            for (int i = 0; i < n; ++i) {
                nodes[i].x += delta(3 * i);
                nodes[i].y += delta(3 * i + 1);
                nodes[i].theta += delta(3 * i + 2);
            }

            std::cout << "Iter " << iter << " Error: " << total_error << "\n";
            if (delta.norm() < 1e-6) break;
        }
    }
};
