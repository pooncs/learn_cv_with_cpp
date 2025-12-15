# Exercise 02: Signals & Slots

## Goal
Connect a `QSlider` to a `QLabel` to display the slider's value in real-time. This demonstrates the core communication mechanism of Qt.

## Learning Objectives
1.  Understand the Observer Pattern implementation in Qt (Signals & Slots).
2.  Connect widgets using `QObject::connect`.
3.  Use lambda expressions as slots.

## Practical Motivation
In a GUI, user actions (clicks, slides) need to trigger updates in other parts of the interface or backend logic. Qt's Signal/Slot mechanism is a type-safe and loose-coupling way to achieve this.

## Theory: Signals & Slots
*   **Signal:** Emitted when an event occurs (e.g., `valueChanged(int)`).
*   **Slot:** A function that reacts to a signal.
*   **Connection:** `connect(sender, &Sender::signal, receiver, &Receiver::slot);`

## Step-by-Step Instructions

### Task 1: Setup UI
Open `todo/src/main.cpp`.
1.  Create a `QWidget` as the main window container.
2.  Create a `QVBoxLayout` and set it on the widget.
3.  Create a `QSlider` (Horizontal) and a `QLabel`.
4.  Add them to the layout.

### Task 2: Connect Signal to Slot
1.  Use `QObject::connect`.
2.  Connect `QSlider::valueChanged` signal.
3.  Connect to a lambda or a slot that updates `QLabel::setText`.
    *   Hint: `label->setNum(value)` or `label->setText(QString::number(value))`.

## Verification
Run the application. Moving the slider should update the number on the label.
