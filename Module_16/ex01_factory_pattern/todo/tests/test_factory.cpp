#include <gtest/gtest.h>
#include "factory.hpp"

class TestAlgo : public IAlgorithm {
public:
    void process() override {}
    std::string name() const override { return "Test"; }
};

TEST(FactoryTest, RegistrationAndCreation) {
    auto& factory = AlgorithmFactory::instance();
    factory.register_algo("Test", []() { return std::make_unique<TestAlgo>(); });
    
    auto algo = factory.create("Test");
    ASSERT_NE(algo, nullptr);
    EXPECT_EQ(algo->name(), "Test");
}

TEST(FactoryTest, UnknownAlgo) {
    auto& factory = AlgorithmFactory::instance();
    EXPECT_THROW(factory.create("NonExistent"), std::runtime_error);
}
