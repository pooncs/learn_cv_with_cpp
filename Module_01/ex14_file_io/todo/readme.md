# Exercise 14: File I/O & Serialization

## Goal
Learn how to read and write data to disk. We will cover **Binary I/O** for raw performance (e.g., image buffers) and **JSON** for human-readable configuration files (using `nlohmann/json`).

## Learning Objectives
1.  **Binary Streams:** `ofstream(..., std::ios::binary)`.
2.  **JSON Serialization:** Mapping C++ structs to JSON objects.
3.  **Third-Party Libraries:** Using `nlohmann/json`.

## Analogy: The Universal Translator vs. The Memory Dump
*   **Binary I/O (The Memory Dump):** You take a photo of your desk and mail it.
    *   *Pros:* Fast, exact copy.
    *   *Cons:* If the receiver has a different desk shape (Endianness/Architecture), the photo doesn't fit.
*   **JSON (The Universal Translator):** You write a list: "1. Pen, 2. Paper".
    *   *Pros:* Everyone understands it. You can edit it with a text editor.
    *   *Cons:* Slower to write/read than just taking a photo.

## Practical Motivation
*   **Configs:** `config.json` -> `{ "camera_id": 0, "threshold": 0.5 }`. Easier to change than recompiling code.
*   **Data:** `features.bin`. Saving 1 million float descriptors as text would be huge and slow. Binary is necessary here.

## Step-by-Step Instructions

### Task 1: Binary I/O
Open `src/main.cpp`.
1.  Create a `std::vector<float> data = {1.1, 2.2, 3.3}`.
2.  Write it to `data.bin` using `std::ofstream`.
    *   Use `write(reinterpret_cast<char*>(ptr), size)`.
3.  Read it back into a new vector and verify values.

### Task 2: JSON Configuration
1.  Include `<nlohmann/json.hpp>`.
2.  Create a JSON object: `json j; j["camera"] = "Logitech"; j["fps"] = 30;`.
3.  Write it to `config.json`.
4.  Read it back: `std::ifstream i("config.json"); json j_in; i >> j_in;`.
5.  Access values: `std::string cam = j_in["camera"];`.

## Verification
Compile and run.
```bash
cd todo
mkdir build && cd build
cmake ..
cmake --build .
./main
```
Output should show matching binary data and correct JSON values.
