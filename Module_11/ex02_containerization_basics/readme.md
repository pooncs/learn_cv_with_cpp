# Exercise 02: Containerization Basics with Docker

## Goal
Containerize a C++ Computer Vision application using Docker.

## Learning Objectives
1.  **Dockerfile:** Learn the syntax (`FROM`, `RUN`, `COPY`, `WORKDIR`, `CMD`).
2.  **Multi-Stage Builds:** Separate the build environment (heavy tools) from the runtime environment (lightweight).
3.  **Dependencies:** Install OpenCV/Eigen inside the container.
4.  **Running:** Build and run the image.

## Practical Motivation
"It works on my machine" is solved by Docker. Shipping a container ensures your CV app runs exactly the same way in production (cloud, edge device) as it does on your dev laptop.

## Theory: Docker Layers
A Docker image is built from layers.
*   **Base Layer:** OS (e.g., Ubuntu 22.04).
*   **Deps Layer:** `apt-get install ...`
*   **Build Layer:** Compile code.
*   **Runtime Layer:** Copy binary and run.

## Step-by-Step Instructions

### Task 1: Create a Simple App
The provided `src/main.cpp` prints "Hello from Docker!".

### Task 2: Write Dockerfile
Create `Dockerfile` in the root of `todo/`.
1.  Use `ubuntu:22.04` as base.
2.  Install `build-essential`, `cmake`, `python3-pip`.
3.  Install `conan`.
4.  Copy source code.
5.  Build the project.
6.  Set entrypoint.

### Task 3: Build and Run
```bash
docker build -t cv_app .
docker run --rm cv_app
```

## Code Hints
*   **Multi-Stage:**
    ```dockerfile
    FROM ubuntu:22.04 AS builder
    # ... install deps and build ...
    
    FROM ubuntu:22.04
    COPY --from=builder /app/build/bin/app /usr/local/bin/app
    CMD ["app"]
    ```

## Verification
The output of `docker run` should be the message from your C++ program.
