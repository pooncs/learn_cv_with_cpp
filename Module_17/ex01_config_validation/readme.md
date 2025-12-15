# Exercise 01: Configuration Validation

## Goal
Implement a robust configuration loader that reads YAML files and validates them against a predefined schema before the application starts. This ensures that the application fails fast and provides meaningful error messages if the configuration is invalid.

## Learning Objectives
1.  **YAML Parsing:** Learn to read and parse YAML files using `yaml-cpp`.
2.  **Schema Validation:** Implement logic to verify that the configuration contains all required fields with correct types and valid ranges.
3.  **Error Handling:** Provide clear, actionable error messages to the user when validation fails.
4.  **Type Safety:** Map dynamic YAML nodes to strongly-typed C++ structures.

## Practical Motivation
In Computer Vision and Machine Learning, experiments are often driven by configuration files (hyperparameters, file paths, model architectures). A common pitfall is starting a long-running training job (e.g., 2 days) only to have it crash halfway through because a parameter was misspelled or a path was missing.

By validating the configuration at startup, we ensure:
-   **Reliability:** The system only runs with valid inputs.
-   **Developer Experience:** Users get immediate feedback on what is wrong with their config.
-   **Reproducibility:** The configuration structure is strictly defined.

## Theory: Configuration & Validation

### 1. YAML for Configuration
YAML (YAML Ain't Markup Language) is widely used for configuration due to its readability.
Example:
```yaml
model:
  name: "YOLOv8"
  input_size: [640, 640]
  confidence_threshold: 0.5
dataset:
  path: "/data/coco"
```

### 2. Validation Strategies
-   **Existence Check:** Ensure required keys are present.
-   **Type Check:** Ensure values are of the expected type (e.g., string, integer, sequence).
-   **Range/Constraint Check:** Ensure values fall within valid bounds (e.g., probability between 0.0 and 1.0).

## Step-by-Step Instructions

### Task 1: Define the Configuration Structure
Open `include/config.hpp`. Define a C++ structure that mirrors the expected YAML configuration.
```cpp
struct ModelConfig {
    std::string name;
    int input_width;
    int input_height;
    float conf_threshold;
};

struct AppConfig {
    ModelConfig model;
    std::string dataset_path;
};
```

### Task 2: Implement the Loader
Open `src/config_loader.cpp`. Use `yaml-cpp` to load the file.
```cpp
YAML::Node config = YAML::LoadFile(path);
```

### Task 3: Implement Validation Logic
Write a function `validate_config(const YAML::Node& node)` that checks:
1.  `model` key exists.
2.  `model.confidence_threshold` is between 0.0 and 1.0.
3.  `dataset.path` is not empty.

Throw a `std::runtime_error` or a custom exception with a descriptive message if validation fails.

### Task 4: Integration
In `src/main.cpp`, call the loader and validator. Catch exceptions and print them using `fmt::print`.

## Code Hints
-   **Checking existence:** `if (node["key"]) { ... }`
-   **Type conversion:** `node["key"].as<int>()`
-   **Iterating sequences:** `for (const auto& item : node["sequence"]) { ... }`

## Common Pitfalls
-   **Missing Keys:** Accessing a missing key like `node["missing"].as<int>()` will throw an exception. It's often better to check existence first or use a default value if appropriate (though for this exercise, we want strict validation).
-   **Type Mismatches:** YAML is loosely typed. "1.0" might be parsed as a string if quoted, or a float if not. Be consistent.

## Verification
Run the tests to ensure your validator correctly identifies valid and invalid configurations.
```bash
cd todo
mkdir build && cd build
conan install .. -s compiler.cppstd=17 --output-folder=. --build=missing
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake
cmake --build .
ctest
```
