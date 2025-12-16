#include <QApplication>
#include <QMainWindow>
#include <QtCharts>
#include <cstdlib>

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    // 1. Create BarSet (e.g., Histogram bins)
    QBarSet *set0 = new QBarSet("Intensity");

    // Add random data (simulating a histogram)
    for (int i = 0; i < 20; ++i) { // 20 bins for simplicity
        *set0 << (rand() % 100);
    }

    // 2. Create Series
    QBarSeries *series = new QBarSeries();
    series->append(set0);

    // 3. Create Chart
    QChart *chart = new QChart();
    chart->addSeries(series);
    chart->setTitle("Simple Histogram");
    chart->setAnimationOptions(QChart::SeriesAnimations);

    // 4. Create Axes
    QBarCategoryAxis *axisX = new QBarCategoryAxis();
    chart->addAxis(axisX, Qt::AlignBottom);
    series->attachAxis(axisX);

    QValueAxis *axisY = new QValueAxis();
    axisY->setRange(0, 100);
    chart->addAxis(axisY, Qt::AlignLeft);
    series->attachAxis(axisY);

    // 5. Create ChartView
    QChartView *chartView = new QChartView(chart);
    chartView->setRenderHint(QPainter::Antialiasing);

    QMainWindow window;
    window.setCentralWidget(chartView);
    window.resize(800, 600);
    window.show();

    return app.exec();
}
