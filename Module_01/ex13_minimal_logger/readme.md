# Exercise 13: Minimal Logger

## Goal
Replace `std::cerr` with a simple leveled logger.

## Learning Objectives
1.  Implement a basic logging system with levels (INFO, WARN, ERROR).
2.  Use a Singleton pattern or static class for global access.
3.  Format output with timestamps (optional) and severity.

## Practical Motivation
`std::cout` is not enough for complex applications. You need to be able to filter logs by severity and direct them to different outputs (console, file).

## Theory & Background

### Log Levels
- **INFO**: General operational entries.
- **WARN**: Something unexpected happened, but execution can continue.
- **ERROR**: Something failed, might need attention.
- **DEBUG**: Detailed information for debugging.

### Singleton Pattern
Ensures a class has only one instance and provides a global point of access.

## Implementation Tasks

### Task 1: Logger Class
Implement `Logger::log(Level, message)`.

### Task 2: Macros
Define macros like `LOG_INFO("msg")` to simplify usage and capture file/line info.

## Common Pitfalls
- Thread safety: `std::cout` is thread-safe for characters but not lines. Multiple threads logging can result in mixed output.
