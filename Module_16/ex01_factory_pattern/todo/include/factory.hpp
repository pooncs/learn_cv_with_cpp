#pragma once
#include <map>
#include <functional>
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>

// Forward Declaration
class IFilter;

class FilterFactory {
public:
    using CreatorFunc = std::function<std::unique_ptr<IFilter>()>;

    // TODO: Implement registerFilter
    // static void registerFilter(const std::string& name, CreatorFunc creator);

    // TODO: Implement createFilter
    // static std::unique_ptr<IFilter> createFilter(const std::string& name);

    // TODO: Implement getAvailableFilters
    // static std::vector<std::string> getAvailableFilters();

private:
    // TODO: Implement Singleton Registry
};
