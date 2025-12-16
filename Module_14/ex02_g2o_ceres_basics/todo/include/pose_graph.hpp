#pragma once
#include <Eigen/Dense>
#include <vector>

struct Pose2D {
    double x, y, theta;
};

struct Edge {
    int from_idx;
    int to_idx;
    Pose2D measurement;
    Eigen::Matrix3d information;
};

class PoseGraphOptimizer {
public:
    static void optimize(std::vector<Pose2D>& nodes, const std::vector<Edge>& edges, int iterations = 10) {
        // TODO: Implement Pose Graph Optimization
        // 1. Loop iterations
        // 2. Build H and b
        // 3. Solve H * delta = -b
        // 4. Update nodes
    }
};
