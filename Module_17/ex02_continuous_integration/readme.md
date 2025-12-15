# Exercise 02: Continuous Integration with GitHub Actions

## Goal
Set up a Continuous Integration (CI) pipeline using GitHub Actions that automatically builds the project and runs tests on every push and pull request.

## Learning Objectives
1.  **CI Concepts:** Understand the role of CI in modern software development.
2.  **GitHub Actions:** Learn the syntax of YAML workflows.
3.  **Automated Testing:** Configure CTest to run within the pipeline.
4.  **Multi-Platform Builds:** (Optional) Configure builds for Ubuntu and Windows.

## Practical Motivation
"It works on my machine" is a classic developer excuse. CI ensures that code works in a clean, isolated environment. It catches compilation errors, missing dependencies, and failing tests *before* code is merged, saving the team from broken builds.

## Theory: GitHub Actions
A GitHub Actions workflow is defined in a YAML file in `.github/workflows/`.
Key components:
-   **on:** Events that trigger the workflow (e.g., `push`, `pull_request`).
-   **jobs:** A set of steps that execute on the same runner.
-   **steps:** Individual tasks (checkout code, install dependencies, build, test).

## Step-by-Step Instructions

### Task 1: Project Setup
The `todo` folder contains a simple C++ project with a test.
1.  Verify you can build and test it locally.
2.  Note the commands you use (cmake, make, ctest).

### Task 2: Create Workflow File
Create a file `.github/workflows/ci.yml`.
Define the trigger:
```yaml
name: CI
on: [push, pull_request]
```

### Task 3: Define the Build Job
1.  Use `ubuntu-latest` as the runner.
2.  Checkout the code using `actions/checkout@v4`.
3.  Install dependencies (CMake, GTest).
    - Note: GitHub runners have CMake pre-installed.
4.  Configure CMake.
5.  Build.
6.  Run Tests.

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Configure CMake
      run: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
      
    - name: Build
      run: cmake --build build --config Release
      
    - name: Test
      run: ctest --test-dir build --output-on-failure
```

### Task 4: Windows Support (Bonus)
Add a matrix strategy to run on `windows-latest` as well.

## Code Hints
-   **Working Directory:** Use `working-directory` or change paths if your CMakeLists.txt is not in the root. For this exercise, assume the root is the exercise folder.
-   **Matrix:**
    ```yaml
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest]
    runs-on: ${{ matrix.os }}
    ```

## Verification
Since you cannot easily run GitHub Actions locally without `act`, the verification for this exercise is to ensure the YAML file is valid and follows best practices. You can validate YAML syntax online.
If you have a GitHub repository, push this code to a branch to see the action run.
