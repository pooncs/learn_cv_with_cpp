# Exercise 01: Factory Pattern

## Goal
Implement a `ModuleFactory` to create algorithms (like Filters) by string name without hardcoding the class types in the main logic.

## Learning Objectives
1.  **Factory Pattern:** Encapsulate object creation.
2.  **Polymorphism:** Use a common interface (`IFilter`) for different implementations.
3.  **Registration:** Learn how to register classes into the factory (static registration vs dynamic).

## Practical Motivation
In a large CV pipeline, you might want to load a configuration file like:
```yaml
pipeline:
  - type: "Blur"
    kernel_size: 5
  - type: "Edge"
    threshold: 100
```
Your code needs to read the string "Blur" and instantiate a `BlurFilter` class. You can't write `if (type == "Blur") new BlurFilter()` for every possible filter, or your code will be a mess of if-else statements. A Factory solves this.

## Step-by-Step Instructions
1.  **Define Interface:** Create `IFilter` with a pure virtual method `process(cv::Mat& img)`.
2.  **Implement Concrete Classes:** Create `BlurFilter` and `EdgeFilter` implementing `IFilter`.
3.  **Create Factory:**
    *   Use a `std::map<std::string, std::function<std::unique_ptr<IFilter>()>>` to store creators.
    *   Implement `registerFilter(name, creator)` and `createFilter(name)`.
4.  **Use it:** Register your filters, then ask the factory to create them by name.

## Todo
1.  Define the `IFilter` interface.
2.  Implement the `ModuleFactory` class.
3.  Register at least two filters.
4.  Demonstrate creating and using them.

## Verification
*   Calling `create("Blur")` should return a valid `BlurFilter`.
*   Calling `create("Unknown")` should return null or throw an error.
