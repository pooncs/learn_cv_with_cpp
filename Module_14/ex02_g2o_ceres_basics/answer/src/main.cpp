#include "pose_graph.hpp"
#include <fmt/core.h>

int main() {
    // 3 Nodes: 0 -> 1 -> 2 -> 0 (Loop closure)
    std::vector<Pose2D> nodes = {
        {0, 0, 0},         // Node 0 (Fixed)
        {1.1, 0.1, 0.1},   // Node 1 (Noisy)
        {1.1, 1.1, 1.57}   // Node 2 (Noisy)
    };

    std::vector<Edge> edges;
    Eigen::Matrix3d info = Eigen::Matrix3d::Identity() * 100.0;

    // Edge 0->1: Move x+1
    edges.push_back({0, 1, {1.0, 0.0, 0.0}, info});

    // Edge 1->2: Move y+1, turn 90 deg
    edges.push_back({1, 2, {0.0, 1.0, 1.5708}, info});

    // Edge 2->0: Loop closure (Move x+1, y-1, turn -90? No. From 2 to 0.)
    // True pos 2 is (1,1, 90deg). To get back to (0,0,0):
    // Relative: Turn -90, move x+1?
    // Let's assume we measured 0->2 directly as (1,1, 90).
    // Or 2->0 relative.
    // Let's add Edge 0->2 directly: x=1, y=1, th=90
    edges.push_back({0, 2, {1.0, 1.0, 1.5708}, info});

    fmt::print("Before Optimization:\n");
    for (int i = 0; i < nodes.size(); ++i) {
        fmt::print("Node {}: x={:.2f} y={:.2f} th={:.2f}\n", i, nodes[i].x, nodes[i].y, nodes[i].theta);
    }

    PoseGraphOptimizer::optimize(nodes, edges);

    fmt::print("After Optimization:\n");
    for (int i = 0; i < nodes.size(); ++i) {
        fmt::print("Node {}: x={:.2f} y={:.2f} th={:.2f}\n", i, nodes[i].x, nodes[i].y, nodes[i].theta);
    }

    return 0;
}
