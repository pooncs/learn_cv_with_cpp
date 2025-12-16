# Exercise 07: File Dialogs

## Goal
Implement "Open Image" and "Save Result" functionality using standard OS file dialogs.

## Learning Objectives
1.  Use `QFileDialog::getOpenFileName` to select files.
2.  Use `QFileDialog::getSaveFileName` to choose save locations.
3.  Filter file extensions (e.g., "Images (*.png *.jpg)").

## Practical Motivation
Hardcoding paths (`data/lenna.png`) is bad. Users need to select their own data.

## Theory: Static Dialog Functions
Qt provides static helper functions that launch native dialogs and return the selected path as a `QString`.
*   `QString path = QFileDialog::getOpenFileName(parent, title, dir, filter);`

## Step-by-Step Instructions

### Task 1: Setup UI with Buttons
Open `todo/src/main.cpp`.
1.  Create a `QWidget` and `QVBoxLayout`.
2.  Create two `QPushButton`: "Open Image" and "Save Text".
3.  Create a `QLabel` to show the selected path.

### Task 2: Implement Open Slot
1.  Connect "Open Image" button to a lambda.
2.  Call `QFileDialog::getOpenFileName`.
3.  If path is not empty, set it on the label.

### Task 3: Implement Save Slot
1.  Connect "Save Text" button.
2.  Call `QFileDialog::getSaveFileName`.
3.  If path is not empty, write a simple text file to that path (using `std::ofstream` or `QFile`).

## Verification
Run the app. Click "Open", select a file, see path. Click "Save", type a name, check if file is created.
