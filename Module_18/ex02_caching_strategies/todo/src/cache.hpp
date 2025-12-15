#pragma once
#include <string>

class CacheManager {
public:
    CacheManager(const std::string& cache_dir);

    // TODO: Implement SHA256 hashing
    std::string compute_hash(const std::string& input);

    // TODO: Check file existence
    bool exists(const std::string& key);

    // TODO: Save data to file
    void save(const std::string& key, const std::string& data);

    // TODO: Load data from file
    std::string load(const std::string& key);
};
