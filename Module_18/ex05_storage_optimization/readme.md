# Exercise 05: Storage Optimization

## Goal
Optimize dataset storage for high-performance reading by implementing sharding and compression.

## Learning Objectives
1.  **Sharding:** Split a million tiny files into a few large "shard" files (like TFRecord or WebDataset) to reduce filesystem overhead.
2.  **Compression:** Apply compression (Zlib/Snappy) to chunks.
3.  **Random Access:** Maintain an index to read specific samples from shards without decompressing the whole file.

## Practical Motivation
Reading 1,000,000 small 10KB images kills filesystem performance (IOPS bottleneck). Packing them into 100MB shards allows sequential reads, which are much faster.

## Step-by-Step Instructions

### Task 1: File Format
Define a simple binary format:
`[Header: Count] [Index: Offset, Size...] [Data Blob]`

### Task 2: Packer
Write a tool `pack_data` that reads a directory of images and writes them into shard files (e.g., `data-001.bin`).

### Task 3: Reader
Implement a `ShardReader` class that can request "Image #42" by looking up the index and reading the bytes.

## Verification
Benchmark reading 1000 images from:
1.  Loose files.
2.  Your shard reader.
The shard reader should be significantly faster on cold cache.
