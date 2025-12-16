#include <gtest/gtest.h>
#include <QApplication>
#include <QtConcurrent>
#include <QFuture>

int simpleTask() {
    return 42;
}

TEST(Threading, ConcurrentRun) {
    int argc = 0;
    if (!QApplication::instance()) {
        new QApplication(argc, nullptr);
    }

    QFuture<int> future = QtConcurrent::run(simpleTask);
    future.waitForFinished();
    
    EXPECT_EQ(future.result(), 42);
}
