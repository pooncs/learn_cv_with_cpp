# Exercise 04: Pipeline Pattern

## Goal
Design a linear pipeline where data flows through a sequence of processors.

## Learning Objectives
1.  **Chain of Responsibility:** Each stage processes data and passes it to the next.
2.  **Interface:** `IProcessor::process(Context& ctx)`.
3.  **Flexibility:** Add/Remove stages dynamically.

## Practical Motivation
CV flows are often pipelines: `Acquire -> Preprocess -> Detect -> Track -> Display`. Hardcoding this loop is rigid. A Pipeline class allows you to compose these stages nicely.

## Step-by-Step Instructions
1.  Define `Pipeline` class holding a `vector<unique_ptr<IProcessor>>`.
2.  Implement `add_stage(processor)`.
3.  Implement `execute(data)`.
4.  Create stages: `Grayscale`, `Blur`, `Edge`.

## Verification
*   Pass an image through. Verify it gets modified by all stages in order.
