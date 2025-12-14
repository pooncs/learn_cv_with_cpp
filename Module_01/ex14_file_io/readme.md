# Exercise 14: File I/O & Serialization

## Goal
Read/write binary buffers and JSON configs using `nlohmann_json`.

## Learning Objectives
1.  Read and write binary files (`std::fstream`).
2.  Serialize C++ structs to JSON using `nlohmann/json`.
3.  Deserialize JSON back to structs.

## Practical Motivation
CV apps need to load configuration (camera parameters, thresholds) from JSON/YAML and save large data (images, features) as binary.

## Theory & Background

### Binary I/O
- `std::ofstream(path, std::ios::binary)`
- `write((char*)ptr, size)`
- `read((char*)ptr, size)`

### JSON
- `nlohmann::json` is a modern C++ JSON library.
- `j = json{{"key", value}}`
- `value = j["key"]`
- `NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Type, member1, member2)` for auto serialization.

## Implementation Tasks

### Task 1: Binary I/O
Write a `std::vector<float>` to a binary file and read it back. Verify data integrity.

### Task 2: JSON Config
Create a `Config` struct (width, height, app_name). Save it to `config.json` and load it back.

## Common Pitfalls
- Endianness (though usually not an issue on same machine).
- Opening file without `std::ios::binary` on Windows (converts `\n` to `\r\n`).
