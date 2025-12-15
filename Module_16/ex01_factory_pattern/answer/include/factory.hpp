#pragma once
#include "algorithms.hpp"
#include <map>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

class AlgorithmFactory {
public:
    using Creator = std::function<std::unique_ptr<IAlgorithm>()>;

    static AlgorithmFactory& instance() {
        static AlgorithmFactory factory;
        return factory;
    }

    void register_algo(const std::string& name, Creator creator) {
        registry_[name] = creator;
    }

    std::unique_ptr<IAlgorithm> create(const std::string& name) {
        auto it = registry_.find(name);
        if (it != registry_.end()) {
            return it->second();
        }
        throw std::runtime_error("Algorithm not registered: " + name);
    }

private:
    std::map<std::string, Creator> registry_;
};
