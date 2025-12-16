#include <gtest/gtest.h>
#include <QApplication>
#include <QtCharts>

TEST(Charts, Creation) {
    int argc = 0;
    if (!QApplication::instance()) {
        new QApplication(argc, nullptr);
    }

    QChart *chart = new QChart();
    QBarSeries *series = new QBarSeries();
    QBarSet *set = new QBarSet("Test");
    
    *set << 10 << 20;
    series->append(set);
    chart->addSeries(series);
    
    EXPECT_EQ(series->barSets().size(), 1);
    EXPECT_EQ(set->count(), 2);
    
    delete chart;
}
