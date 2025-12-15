#pragma once
#include <mutex>

namespace cv_curriculum {

class Counter {
public:
    virtual void increment() = 0;
    virtual int get() const = 0;
    virtual ~Counter() = default;
};

class UnsafeCounter : public Counter {
    int value = 0;
public:
    void increment() override {
        value++;
    }
    int get() const override {
        return value;
    }
};

class SafeCounter : public Counter {
    int value = 0;
    mutable std::mutex mtx;
public:
    void increment() override {
        std::lock_guard<std::mutex> lock(mtx);
        value++;
    }
    int get() const override {
        std::lock_guard<std::mutex> lock(mtx);
        return value;
    }
};

} // namespace cv_curriculum
