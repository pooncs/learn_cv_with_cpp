# Exercise 03: Observer Pattern (Event Bus)

## Goal
Implement an Event Bus where "Publishers" can emit events and "Subscribers" can listen to them without direct coupling.

## Learning Objectives
1.  **Decoupling:** Modules don't need to know about each other.
2.  **Subject/Observer:** The classic pattern structure.
3.  **Type Erasure/Templates:** Handling different event payload types (e.g., `ImageEvent`, `PoseEvent`).

## Practical Motivation
The `CameraDriver` captures an image. The `Gui` needs to show it. The `Logger` needs to save it. The `Tracker` needs to process it. Instead of the Camera calling all three, it just publishes `NewFrame`, and others listen.

## Step-by-Step Instructions
1.  Create `EventBus`.
2.  `subscribe<EventType>(callback)`
3.  `publish<EventType>(event)`
4.  Test with a mock `Camera` publishing an `Image` and multiple listeners.

## Verification
*   When one event is published, all registered callbacks should fire.
