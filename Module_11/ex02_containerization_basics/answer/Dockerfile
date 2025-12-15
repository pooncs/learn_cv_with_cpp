# Use Ubuntu as base image
FROM ubuntu:22.04 AS builder

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Conan
RUN pip3 install conan

# Set up Conan profile
RUN conan profile detect --force

# Create working directory
WORKDIR /app

# Copy dependency file
COPY conanfile.txt .

# Install dependencies
RUN conan install . --output-folder=build --build=missing --settings=build_type=Release

# Copy source code
COPY CMakeLists.txt .
COPY src ./src

# Build
WORKDIR /app/build
RUN cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release
RUN cmake --build . --config Release

# Runtime Stage
FROM ubuntu:22.04

# Copy executable from builder stage
COPY --from=builder /app/build/docker_app /usr/local/bin/docker_app

# Set entrypoint
CMD ["docker_app"]
