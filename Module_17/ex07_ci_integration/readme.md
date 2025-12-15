# Exercise 07: Comprehensive CI Integration

## Goal
Combine all quality checks (Linting, Sanitizers, Testing, Coverage) into a single, robust CI pipeline.

## Learning Objectives
1.  **Pipeline Orchestration:** Manage multiple jobs/steps in GitHub Actions.
2.  **Quality Gates:** Fail the build if any check fails (e.g., coverage < 80%).
3.  **Artifacts:** Upload reports (coverage HTML, benchmark JSON) for later inspection.

## Practical Motivation
Running checks manually is tedious. A unified pipeline ensures that every code change meets all quality standards automatically.

## Step-by-Step Instructions

### Task 1: Create Workflow
Edit `.github/workflows/quality.yml`.

### Task 2: Add Jobs
1.  **Lint:** Run `clang-tidy`.
2.  **Sanitize:** Build and run tests with ASan.
3.  **Test & Cover:** Run tests and upload coverage.
4.  **Benchmark:** Run benchmarks and check for regressions.

### Task 3: Upload Artifacts
Use `actions/upload-artifact@v4` to save the coverage report and benchmark results.

```yaml
    - name: Upload Coverage
      uses: actions/upload-artifact@v4
      with:
        name: coverage-report
        path: out/
```

## Verification
Push a commit that breaks a rule (e.g., introduces a memory leak). Verify that the specific CI job fails.
