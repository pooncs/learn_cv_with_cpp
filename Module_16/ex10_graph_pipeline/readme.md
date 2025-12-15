# Exercise 10: Graph Pipeline (DAG)

## Goal
Implement a pipeline where tasks are nodes in a Directed Acyclic Graph (DAG), allowing parallel execution of non-dependent branches.

## Learning Objectives
1.  **Dependencies:** Node B needs output of Node A.
2.  **Topological Sort:** Determining execution order.
3.  **Parallelism:** Executing independent nodes (e.g., C and D depend on A) in parallel.

## Practical Motivation
Linear pipelines are simple, but sometimes you can do "Face Detection" and "Background Subtraction" at the same time before merging them. A DAG executor handles this.

## Step-by-Step Instructions
1.  Define `Node` with inputs/outputs.
2.  Connect nodes: `A -> B`, `A -> C`, `B -> D`, `C -> D`.
3.  Implement `Executor`.
    *   Find nodes with 0 unsatisfied dependencies.
    *   Run them (potentially in threads).
    *   Mark complete, update dependents.
    *   Repeat.

## Verification
*   Construct a diamond graph (A->B, A->C, B->D, C->D).
*   Add print statements with delays.
*   Verify B and C run in parallel (timestamps overlap).
