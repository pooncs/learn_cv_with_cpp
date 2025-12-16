# Exercise 02: g2o Basics

## Goal
Use the g2o (General Graph Optimization) library to solve a simple graph optimization problem.

## Learning Objectives
1.  **Graph Optimization:** Representing problems as nodes (variables) and edges (constraints).
2.  **g2o Types:** `VertexSE3`, `EdgeSE3`, `BlockSolver`, `LinearSolver`.
3.  **Building the Graph:** Adding vertices and edges.
4.  **Optimizing:** Running `optimizer.optimize()`.

## Practical Motivation
SLAM is "Graph Optimization". The robot poses are nodes. The odometry measurements are edges between consecutive nodes. Loop closures are edges between non-consecutive nodes. g2o minimizes the total error in the graph.

**Analogy:**
*   **Nodes:** Pins on a map (Cities).
*   **Edges:** Rubber bands connecting them (Roads with measured distances).
*   **Optimization:** Letting the rubber bands pull the pins until they settle into a state of minimum tension (Minimum Energy/Error).

## Step-by-Step Instructions

### Task 1: Setup Optimizer
1.  Create `LinearSolver` (Dense or Sparse).
2.  Create `BlockSolver`.
3.  Create `OptimizationAlgorithm` (Levenberg-Marquardt or Gauss-Newton).
4.  Assign algorithm to `g2o::SparseOptimizer`.

### Task 2: Build Graph (Curve Fitting)
*   *Note: For simplicity, we can do curve fitting (Vertex = parameters a,b,c; Edge = data point) or Pose Graph Optimization.*
*   **Curve Fitting Scenario:**
    1.  Define a custom Vertex (stores $a, b, c$).
    2.  Define a custom Edge (computes error $y - \exp(ax^2 + bx + c)$).
    3.  Add vertex with initial guess.
    4.  Add edges for noisy data points.

## Code Hints
```cpp
// 1. Solver
auto linearSolver = std::make_unique<g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>>();
auto solver = std::make_unique<BlockSolverType>(std::move(linearSolver));
auto algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(solver));
optimizer.setAlgorithm(algorithm);

// 2. Vertex
CurveFittingVertex* v = new CurveFittingVertex();
v->setEstimate(Eigen::Vector3d(0,0,0));
v->setId(0);
optimizer.addVertex(v);

// 3. Edge
CurveFittingEdge* edge = new CurveFittingEdge(x_data);
edge->setMeasurement(y_data);
edge->setInformation(Eigen::Matrix<double,1,1>::Identity());
edge->setVertex(0, v);
optimizer.addEdge(edge);
```

## Verification
The optimized parameters should match the ground truth used to generate the data.
