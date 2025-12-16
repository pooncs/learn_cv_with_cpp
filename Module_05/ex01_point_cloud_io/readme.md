# Module 05 - Exercise 01: Point Cloud IO (Manual)

## Goal
Understand the structure of 3D point cloud file formats by manually implementing readers and writers for **XYZ** and **PLY** files.

## Concept: Point Clouds as Data Tables
Imagine a 3D point cloud as a simple spreadsheet.
- Each **row** represents a single point in space.
- The columns are `X`, `Y`, and `Z` coordinates.
- Sometimes there are extra columns for **Color** (R, G, B) or **Normals** (Nx, Ny, Nz).

### The Formats
1.  **XYZ**: The simplest format. Just text lines with numbers.
    ```
    1.0 2.0 3.0
    4.5 5.5 6.5
    ...
    ```
2.  **PLY (Polygon File Format)**: A more structured format with a **Header** that describes the data, followed by the data itself (ASCII or Binary). We will focus on ASCII.
    ```
    ply
    format ascii 1.0
    element vertex 10
    property float x
    property float y
    property float z
    end_header
    1.0 2.0 3.0
    ...
    ```

## Task
1.  Implement a `struct Point3D { float x, y, z; };`
2.  Write a function `writeXYZ` to save a vector of points to a file.
3.  Write a function `readXYZ` to load points from a file.
4.  Write a function `writePLY` that includes a basic header.
5.  (Optional) Verify by opening the file in a viewer like MeshLab or CloudCompare (or just checking text).

## Instructions
1.  Navigate to `todo/` directory.
2.  Open `src/main.cpp`.
3.  Implement the functions where indicated by `// TODO`.
4.  Build and run.

## Build
```bash
mkdir build
cd build
cmake ..
cmake --build .
```
