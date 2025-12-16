# Exercise 09: Charts

## Goal
Plot a live histogram using Qt Charts.

## Learning Objectives
1.  Use `QChart`, `QChartView`, and `QBarSeries`.
2.  Compute histogram from an image.
3.  Update the chart dynamically.

## Practical Motivation
Visualizing data distributions (histograms) is crucial for understanding image contrast and brightness.

## Theory: Qt Charts
*   **QChart:** The chart itself (title, legend).
*   **QChartView:** The widget that displays the chart.
*   **QBarSeries / QLineSeries:** The data container.
*   **QBarSet:** A set of bars (e.g., one for each bin).

## Step-by-Step Instructions

### Task 1: Setup
Open `todo/src/main.cpp`.
1.  Include `<QtCharts>`.
2.  Create a `QChartView`.

### Task 2: Create Histogram Data
1.  Create a `QBarSet("Intensity")`.
2.  Fill it with dummy data or real histogram data (256 bins).
    *   For simplicity, just add random values: `set->append(rand() % 100);`

### Task 3: Build Chart
1.  Create `QBarSeries` and append the set.
2.  Create `QChart`, add series.
3.  Create `QBarCategoryAxis` (optional, for X labels).
4.  Create `QValueAxis` (for Y height).
5.  Set chart on `QChartView`.

## Verification
Run the app. A bar chart with 256 bars should appear.
