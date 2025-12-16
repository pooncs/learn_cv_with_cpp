# Exercise 02: Containerization with Docker

## Goal
Package a C++ application into a Docker container.

## Learning Objectives
1.  **Dockerfile:** Write a multi-stage `Dockerfile` for C++.
2.  **Building:** Use `docker build` to create an image.
3.  **Running:** Use `docker run` to execute the container.
4.  **Minimization:** Understand the difference between a "Build" stage (heavy, with compilers) and a "Runtime" stage (light, just the binary).

## Practical Motivation
"It works on my machine" is the most common developer excuse. Docker ensures that if it runs on your machine (in a container), it runs on the server, your colleague's machine, and the cloud.

**Analogy:**
*   **Without Docker:** Moving a house by carrying every piece of furniture, appliance, and brick separately. You might forget the toaster, or the new house might have different power outlets.
*   **With Docker:** Putting the entire house inside a standardized shipping container. You just move the container. It doesn't matter if the destination is a ship, a truck, or a train; the contents remain exactly as you arranged them.

## Theory: Multi-Stage Builds
C++ binaries don't need `gcc`, `cmake`, or source code to run. They only need the OS libraries (libc).
1.  **Stage 1 (Builder):** Has `gcc`, `cmake`, dependencies. Compiles source -> `my_app`.
2.  **Stage 2 (Runtime):** Minimal OS (e.g., `alpine` or `distroless`). Copy `my_app` from Stage 1. Result: Tiny image.

## Step-by-Step Instructions

### Task 1: C++ App
1.  Write a simple `main.cpp` that prints "Hello from Docker!".

### Task 2: Dockerfile
1.  Create a `Dockerfile`.
2.  **Stage 1:** From `ubuntu:22.04`. Install `build-essential`, `cmake`. Copy source. Build.
3.  **Stage 2:** From `ubuntu:22.04` (or smaller). Copy binary from Stage 1. Set `ENTRYPOINT`.

### Task 3: Build & Run
1.  `docker build -t my_cpp_app .`
2.  `docker run --rm my_cpp_app`

## Code Hints
```dockerfile
# Build Stage
FROM ubuntu:22.04 AS builder
RUN apt-get update && apt-get install -y build-essential cmake
WORKDIR /app
COPY . .
RUN cmake . && make

# Runtime Stage
FROM ubuntu:22.04
WORKDIR /app
COPY --from=builder /app/my_app .
CMD ["./my_app"]
```

## Verification
Running the container should output your message.
