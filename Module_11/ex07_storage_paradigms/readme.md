# Exercise 07: Storage Paradigms

## Goal
Compare storing images in a filesystem (Data Lake) vs a Database (Data Warehouse/Blob Store).

## Learning Objectives
1.  **File I/O:** Reading thousands of small files.
2.  **SQLite/Parquet:** Storing binary blobs or metadata.
3.  **Performance:** Benchmark read speeds.

## Practical Motivation
Millions of small files kill filesystem performance (inode lookup). Aggregated formats are faster.

## Step-by-Step Instructions
1.  Write 1000 tiny images to disk.
2.  Write 1000 tiny images to a single SQLite DB.
3.  Measure time to read all back.

## Verification
DB/Aggregated read should be faster on cold cache.
