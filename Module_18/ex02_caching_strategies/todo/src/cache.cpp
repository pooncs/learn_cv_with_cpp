#include "cache.hpp"

CacheManager::CacheManager(const std::string& cache_dir) {
}

std::string CacheManager::compute_hash(const std::string& input) {
    return "";
}

bool CacheManager::exists(const std::string& key) {
    return false;
}

void CacheManager::save(const std::string& key, const std::string& data) {
}

std::string CacheManager::load(const std::string& key) {
    return "";
}
