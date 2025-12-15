# Exercise 09: Logging System

## Goal
Implement a thread-safe, leveled logging system.

## Learning Objectives
1.  **Levels:** DEBUG, INFO, WARN, ERROR.
2.  **Thread Safety:** Multiple threads writing to `std::cout` simultaneously causes garbled output.
3.  **Sinks:** Writing to console vs file.

## Practical Motivation
`printf` debugging is not enough for complex multi-threaded apps. You need timestamps, levels, and thread IDs to debug race conditions.

## Step-by-Step Instructions
1.  Singleton `Logger` class.
2.  `log(level, message)`.
3.  Use `std::mutex` to lock output.
4.  Add timestamp and thread ID to the message.

## Verification
*   Spawn 10 threads, all logging furiously.
*   Output should be clean (no interleaved lines).
