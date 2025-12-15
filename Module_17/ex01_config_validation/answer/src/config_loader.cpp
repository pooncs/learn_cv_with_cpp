#include "config.hpp"
#include <stdexcept>
#include <fmt/core.h>

void ConfigLoader::validate(const YAML::Node& node) {
    if (!node["model"]) throw std::runtime_error("Missing 'model' section");
    if (!node["dataset"]) throw std::runtime_error("Missing 'dataset' section");

    auto model = node["model"];
    if (!model["name"]) throw std::runtime_error("Missing 'model.name'");
    if (!model["input_size"]) throw std::runtime_error("Missing 'model.input_size'");
    if (!model["confidence_threshold"]) throw std::runtime_error("Missing 'model.confidence_threshold'");

    if (!model["input_size"].IsSequence() || model["input_size"].size() != 2) {
        throw std::runtime_error("'model.input_size' must be a sequence of 2 integers");
    }

    float conf = model["confidence_threshold"].as<float>();
    if (conf < 0.0f || conf > 1.0f) {
        throw std::runtime_error(fmt::format("'model.confidence_threshold' must be between 0.0 and 1.0, got {}", conf));
    }

    auto dataset = node["dataset"];
    if (!dataset["path"]) throw std::runtime_error("Missing 'dataset.path'");
}

AppConfig ConfigLoader::load(const std::string& path) {
    try {
        YAML::Node node = YAML::LoadFile(path);
        validate(node);

        AppConfig config;
        auto model = node["model"];
        config.model.name = model["name"].as<std::string>();
        config.model.input_size = model["input_size"].as<std::vector<int>>();
        config.model.confidence_threshold = model["confidence_threshold"].as<float>();
        
        config.dataset_path = node["dataset"]["path"].as<std::string>();
        
        return config;
    } catch (const YAML::BadFile& e) {
        throw std::runtime_error(fmt::format("Could not open file: {}", path));
    } catch (const YAML::Exception& e) {
        throw std::runtime_error(fmt::format("YAML parsing error: {}", e.what()));
    }
}
