# Exercise 06: Big Data Handling

## Goal
Simulate processing a large dataset using a "MapReduce" style approach in C++.

## Learning Objectives
1.  **Sharding:** Split data into chunks.
2.  **Parallel Processing:** Process chunks in threads.
3.  **Aggregation:** Combine results.

## Practical Motivation
Processing 1PB of images requires distributed systems, but the logic starts with parallel processing on one node.

## Step-by-Step Instructions
1.  Create a list of 1000 "files" (strings).
2.  Split into 4 shards.
3.  Launch 4 threads to count characters in each shard.
4.  Sum the totals.

## Verification
The parallel sum must equal the sequential sum.
