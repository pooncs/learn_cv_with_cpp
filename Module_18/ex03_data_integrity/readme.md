# Exercise 03: Data Integrity

## Goal
Ensure data has not been corrupted during transfer or storage by validating checksums.

## Learning Objectives
1.  **Checksums:** Calculate MD5/SHA-256 of files.
2.  **Validation:** Verify files against a manifest.
3.  **Error Handling:** Corrupt a file and detect it.

## Practical Motivation
A flipped bit in a large binary dataset can cause mysterious crashes or training divergence. Validating integrity before processing is essential.

## Step-by-Step Instructions

### Task 1: Checksum Utility
Implement `std::string calculate_sha256(const std::string& filepath)`.

### Task 2: Manifest Format
Define a JSON format mapping filenames to hashes.
```json
{
  "data/img1.png": "a1b2...",
  "data/img2.png": "c3d4..."
}
```

### Task 3: Validator
Write a tool that reads the manifest and checks every file. Report any mismatches or missing files.

## Verification
1.  Generate a manifest for a folder.
2.  Modify one byte in a file using a hex editor.
3.  Run the validator -> It must report the corruption.
