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
        // TODO: Implement unsynchronized increment
    }
    int get() const override {
        return value;
    }
};

class SafeCounter : public Counter {
    int value = 0;
    // TODO: Add mutex
public:
    void increment() override {
        // TODO: Implement synchronized increment
    }
    int get() const override {
        return value; // Should lock too?
    }
};

} // namespace cv_curriculum
