# Build Stage
FROM ubuntu:22.04 AS builder

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy source code
COPY . .

# Build
RUN cmake . && make

# Runtime Stage
FROM ubuntu:22.04

# Create a non-root user for security
RUN useradd -m appuser
USER appuser

WORKDIR /app

# Copy only the artifact from the builder stage
COPY --from=builder /app/docker_demo .

# Run the application
ENTRYPOINT ["./docker_demo"]
