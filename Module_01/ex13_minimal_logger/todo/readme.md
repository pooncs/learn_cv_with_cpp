# Exercise 13: Minimal Logger

## Goal
Replace scattered `std::cerr` / `std::cout` calls with a centralized, leveled Logger class. This allows you to filter messages by importance (e.g., show ERRORs but hide DEBUGs).

## Learning Objectives
1.  **Singleton Pattern (Simplified):** A single global access point for logging (or static methods).
2.  **Enum Classes:** Type-safe log levels.
3.  **Conditional Output:** Printing only if `level >= current_level`.

## Analogy: The Radio Dispatch
*   **Old C++ (`cout`/`cerr`):** Everyone shouts in the room.
    *   "Error: Fire!"
    *   "Debug: I tied my shoe."
    *   "Info: Door opened."
    *   *Result:* Noise. You miss the fire alarm because someone is talking about shoes.
*   **Modern C++ (Logger):** You have a **Radio** with channels.
    *   You set your radio to "WARNINGS ONLY".
    *   You hear "Error: Fire!".
    *   You *don't* hear "Debug: I tied my shoe."
    *   Peace and clarity.

## Practical Motivation
In a CV application running on a robot:
*   **Development:** You want to see "Detected 5 features", "Matrix size 3x3".
*   **Production:** You only want to see "CAMERA DISCONNECTED".
*   If you use `cout`, you have to delete lines manually. With a Logger, you just change one line: `Logger::setLevel(LogLevel::ERROR)`.

## Step-by-Step Instructions

### Task 1: Define Log Levels
Open `src/main.cpp`. Create an `enum class LogLevel { DEBUG, INFO, WARNING, ERROR };`.

### Task 2: The Logger Class
Create a class `Logger`.
*   Static member `current_level`.
*   Static method `setLevel(LogLevel level)`.
*   Static method `log(LogLevel level, std::string message)`.

### Task 3: Implementation
Inside `log()`:
*   If `level < current_level`, return (ignore).
*   Otherwise, print `[LEVEL] message`.
    *   Tip: Map enum to string like "[INFO]" or "[ERROR]".

### Task 4: Usage
In `main()`:
1.  Set level to `INFO`.
2.  Log a DEBUG message (Should NOT appear).
3.  Log an ERROR message (Should appear).
4.  Change level to `DEBUG`.
5.  Log a DEBUG message (Should appear).

## Verification
Compile and run.
```bash
cd todo
mkdir build && cd build
cmake ..
cmake --build .
./main
```
Output should reflect the filtering logic.
