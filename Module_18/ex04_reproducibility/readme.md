# Exercise 04: Experiment Reproducibility

## Goal
Automatically generate a "Run Manifest" for every execution, capturing the exact code version, configuration, and data hash used.

## Learning Objectives
1.  **Metadata Capture:** Record Git commit hash, command line args, and timestamp.
2.  **Config Snapshot:** Save the exact config used.
3.  **Traceability:** Link output artifacts to the run manifest.

## Practical Motivation
"I got 99% accuracy last week but I can't remember which parameter I changed." This exercise solves that.

## Step-by-Step Instructions

### Task 1: Git Info
Use a library or system command (`git rev-parse HEAD`) to get the current commit hash at runtime.

### Task 2: Manifest Generation
Create a class `ExperimentTracker`. On startup:
-   Create a directory `runs/YYYY-MM-DD_HH-MM-SS/`.
-   Save `manifest.json` containing:
    -   `git_commit`
    -   `config` (dump of params)
    -   `data_hash` (checksum of input)

### Task 3: Output Redirection
Save logs and results into that specific run directory.

## Verification
Run the program. Check the `runs/` folder. You should see a new directory with a JSON file describing exactly how to reproduce that run.
