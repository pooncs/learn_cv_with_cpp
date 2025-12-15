#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <filesystem>
#include <openssl/sha.h>
#include <iomanip>
#include <sstream>

namespace fs = std::filesystem;

class CacheManager {
public:
    CacheManager(const std::string& cache_dir) : dir_(cache_dir) {
        if (!fs::exists(dir_)) fs::create_directories(dir_);
    }

    std::string compute_hash(const std::string& input) {
        unsigned char hash[SHA256_DIGEST_LENGTH];
        SHA256(reinterpret_cast<const unsigned char*>(input.c_str()), input.size(), hash);
        std::stringstream ss;
        for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
            ss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
        }
        return ss.str();
    }

    bool exists(const std::string& key) {
        return fs::exists(dir_ / key);
    }

    void save(const std::string& key, const std::string& data) {
        std::ofstream out(dir_ / key);
        out << data;
    }

    std::string load(const std::string& key) {
        std::ifstream in(dir_ / key);
        std::string str((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        return str;
    }

private:
    fs::path dir_;
};
