# Exercise 01: Dataset Versioning

## Goal
Manage large binary datasets (images, videos) alongside code using tools like DVC (Data Version Control).

## Learning Objectives
1.  **Code vs Data:** Understand why Git is bad for large binary files.
2.  **Pointer Files:** How DVC replaces large files with small text pointers.
3.  **Remote Storage:** Pushing/pulling data to S3, Google Drive, or local shared folders.

## Practical Motivation
In Machine Learning and CV, data changes just like code. If you train a model on "Dataset v1" and later "Dataset v2" causes a regression, you need to be able to go back to "Dataset v1" exactly.
**Analogy:** Git is for code (text). DVC is for data (binary). Putting a 50GB video in Git is like trying to stuff a mattress into a standard letter envelope. It tears the envelope and is impossible to mail. DVC puts the mattress in a warehouse (Cloud Storage) and puts a receipt with the location (Pointer file) in the envelope. You mail the envelope (Git), and the recipient uses the receipt to get the mattress.

## Step-by-Step Instructions
1.  **Initialize DVC:** `dvc init` in your repo.
2.  **Add Data:** `dvc add data/images.zip`. This creates `data/images.zip.dvc`.
3.  **Track in Git:** `git add data/images.zip.dvc .gitignore`.
4.  **Push Data:** `dvc push` (requires remote config).
5.  **Pull Data:** `dvc pull` to retrieve the large file based on the pointer.

## Todo
1.  Install DVC (pip or choco).
2.  Initialize a dummy folder with some images.
3.  Run the DVC commands to version it.

## Verification
*   Delete the actual image file.
*   Run `dvc pull`.
*   The file should reappear.
