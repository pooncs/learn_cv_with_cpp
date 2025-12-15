# Exercise 01: Point Cloud IO

## Goal
Read and Write PLY and XYZ files manually to understand 3D data formats.

## Learning Objectives
1.  Parse ASCII PLY/XYZ files.
2.  Store points in a `std::vector<Point3D>`.
3.  Write data back to file.

## Theory & Background

### Point Cloud
A set of data points in space. $P_i = (x, y, z)$.
Often includes color $(r, g, b)$ or normal $(nx, ny, nz)$.

### File Formats
- **XYZ**: Simple ASCII, one point per line: `x y z`.
- **PLY**: Polygon File Format. Header + Data.
  ```
  ply
  format ascii 1.0
  element vertex 100
  property float x
  property float y
  property float z
  end_header
  0.1 0.2 0.3
  ...
  ```

## Implementation Tasks

### Task 1: Point Struct
Define `struct Point3D { float x, y, z; };`.

### Task 2: Read/Write XYZ
Implement `read_xyz` and `write_xyz`.

### Task 3: Read/Write PLY
Implement basic ASCII PLY support (header parsing).

## Common Pitfalls
- Handling different line endings (\r\n vs \n).
- Parsing header correctly.
