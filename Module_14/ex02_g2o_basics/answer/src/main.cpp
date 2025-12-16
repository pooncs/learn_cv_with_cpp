#include <iostream>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <Eigen/Core>
#include <cmath>
#include <vector>

// Curve Model: y = exp(a*x^2 + b*x + c)
// Vertex: Stores parameters [a, b, c]
class CurveFittingVertex : public g2o::BaseVertex<3, Eigen::Vector3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    virtual void setToOriginImpl() override {
        _estimate << 0, 0, 0;
    }

    virtual void oplusImpl(const double* update) override {
        _estimate += Eigen::Vector3d(update);
    }

    virtual bool read(std::istream& in) override { return true; }
    virtual bool write(std::ostream& out) const override { return true; }
};

// Edge: Connects vertex to a data point (x, y)
// Measurement: y
// Error: y_measured - y_model
class CurveFittingEdge : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double _x;

    CurveFittingEdge(double x) : BaseUnaryEdge(), _x(x) {}

    virtual void computeError() override {
        const CurveFittingVertex* v = static_cast<const CurveFittingVertex*>(_vertices[0]);
        const Eigen::Vector3d abc = v->estimate();
        // Model: exp(ax^2 + bx + c)
        double y_model = std::exp(abc[0]*_x*_x + abc[1]*_x + abc[2]);
        _error[0] = _measurement - y_model;
    }

    // Jacobian (Optional but recommended for speed)
    // d(error) / d(abc)
    virtual void linearizeOplus() override {
        const CurveFittingVertex* v = static_cast<const CurveFittingVertex*>(_vertices[0]);
        const Eigen::Vector3d abc = v->estimate();
        double y_model = std::exp(abc[0]*_x*_x + abc[1]*_x + abc[2]);

        _jacobianOplusXi[0] = -_x * _x * y_model; // da
        _jacobianOplusXi[1] = -_x * y_model;      // db
        _jacobianOplusXi[2] = -y_model;           // dc
    }

    virtual bool read(std::istream& in) override { return true; }
    virtual bool write(std::ostream& out) const override { return true; }
};

int main() {
    // 1. Generate Data
    double a = 1.0, b = 2.0, c = 1.0;
    int N = 100;
    double w_sigma = 1.0;
    std::vector<double> x_data, y_data;
    
    std::cout << "Ground Truth: " << a << " " << b << " " << c << std::endl;

    for (int i = 0; i < N; ++i) {
        double x = i / 100.0;
        x_data.push_back(x);
        double noise = 0.0; // Simplify for deterministic run, or add rand
        y_data.push_back(std::exp(a*x*x + b*x + c) + noise);
    }

    // 2. Setup g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>> BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>())
    );

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    // 3. Add Vertex
    CurveFittingVertex* v = new CurveFittingVertex();
    v->setEstimate(Eigen::Vector3d(0, 0, 0)); // Initial Guess
    v->setId(0);
    optimizer.addVertex(v);

    // 4. Add Edges
    for (int i = 0; i < N; ++i) {
        CurveFittingEdge* edge = new CurveFittingEdge(x_data[i]);
        edge->setId(i + 1);
        edge->setVertex(0, v);
        edge->setMeasurement(y_data[i]);
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1.0 / (w_sigma * w_sigma));
        optimizer.addEdge(edge);
    }

    // 5. Optimize
    std::cout << "Start Optimization..." << std::endl;
    optimizer.initializeOptimization();
    optimizer.optimize(10);

    Eigen::Vector3d est = v->estimate();
    std::cout << "Estimated: " << est.transpose() << std::endl;

    return 0;
}
