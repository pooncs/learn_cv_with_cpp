#pragma once
#include <string>
#include <map>
#include <functional>
#include <memory>
#include <stdexcept>

class IAlgorithm {
public:
    virtual ~IAlgorithm() = default;
    virtual void process() = 0;
    virtual std::string name() const = 0;
};

class AlgorithmFactory {
public:
    using Creator = std::function<std::unique_ptr<IAlgorithm>()>;

    static AlgorithmFactory& instance() {
        static AlgorithmFactory factory;
        return factory;
    }

    void register_algo(const std::string& name, Creator creator) {
        // TODO: Store creator in map
    }

    std::unique_ptr<IAlgorithm> create(const std::string& name) {
        // TODO: Look up creator and return new instance
        // Throw exception if not found
        return nullptr;
    }

private:
    std::map<std::string, Creator> registry_;
};
