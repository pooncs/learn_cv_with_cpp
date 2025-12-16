#pragma once
#include "algorithms.hpp"
#include <map>
#include <functional>
#include <memory>
#include <string>
#include <stdexcept>

class FilterFactory {
public:
    using CreatorFunc = std::function<std::unique_ptr<IFilter>()>;

    // Register a new filter type
    static void registerFilter(const std::string& name, CreatorFunc creator) {
        getRegistry()[name] = creator;
    }

    // Create a filter by name
    static std::unique_ptr<IFilter> createFilter(const std::string& name) {
        auto& reg = getRegistry();
        auto it = reg.find(name);
        if (it != reg.end()) {
            return it->second();
        }
        throw std::runtime_error("Unknown filter type: " + name);
    }

    // List available filters
    static std::vector<std::string> getAvailableFilters() {
        std::vector<std::string> keys;
        for (const auto& kv : getRegistry()) {
            keys.push_back(kv.first);
        }
        return keys;
    }

private:
    // Static registry (Meyers Singleton to ensure initialization)
    static std::map<std::string, CreatorFunc>& getRegistry() {
        static std::map<std::string, CreatorFunc> registry;
        return registry;
    }
};
