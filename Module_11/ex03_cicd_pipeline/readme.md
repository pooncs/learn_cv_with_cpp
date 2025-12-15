# Exercise 03: CI/CD Pipeline

## Goal
Set up a Continuous Integration pipeline using GitHub Actions.

## Learning Objectives
1.  **YAML Workflow:** Define steps for checkout, build, and test.
2.  **Matrix Builds:** Test on Ubuntu and Windows.
3.  **Artifacts:** Upload build outputs.

## Practical Motivation
Automate testing to catch bugs early.

## Step-by-Step Instructions
1.  Create `.github/workflows/ci.yml`.
2.  Define a job `build`.
3.  Steps: Checkout -> Install Conan -> CMake Configure -> Build -> Test.

## Verification
Push to GitHub and check the Actions tab.
