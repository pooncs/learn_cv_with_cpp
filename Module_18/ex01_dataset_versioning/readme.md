# Exercise 01: Dataset Versioning

## Goal
Manage large datasets (images, weights) using DVC (Data Version Control) or Git LFS, keeping them coupled with code versioning but stored efficiently.

## Learning Objectives
1.  **Large Files:** Understand why Git is bad for large binaries.
2.  **DVC Basics:** Init, add, push, and pull data.
3.  **Remote Storage:** Configure a local or cloud remote.
4.  **Reproducibility:** Link a specific data version to a git commit.

## Practical Motivation
You trained a model on "dataset_v1". A month later, you want to reproduce it, but someone overwrote the folder with "dataset_v2". DVC solves this by versioning data just like code.

## Step-by-Step Instructions

### Task 1: Setup DVC
Initialize DVC in the `todo` folder (or a new repo).
```bash
dvc init
```

### Task 2: Track Data
Add a large file (simulate with a dummy file).
```bash
dvc add data/images.tar.gz
```
This creates `data/images.tar.gz.dvc` and adds `data/images.tar.gz` to `.gitignore`.

### Task 3: Commit
Commit the `.dvc` file to Git.
```bash
git add data/images.tar.gz.dvc .gitignore
git commit -m "Add dataset v1"
```

### Task 4: Modify and Version
Update the dataset. Run `dvc add` again. Commit the new `.dvc` file.
Now you can switch between data versions using `git checkout` and `dvc checkout`.

## Verification
1.  Checkout the previous git commit.
2.  Run `dvc checkout`.
3.  Verify the data file content matches the old version.
