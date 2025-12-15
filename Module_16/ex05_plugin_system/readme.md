# Exercise 05: Plugin System

## Goal
Load C++ code (shared libraries) at runtime to extend functionality.

## Learning Objectives
1.  **Dynamic Loading:** `dlopen`/`dlsym` (Linux) or `LoadLibrary`/`GetProcAddress` (Windows).
2.  **ABI Stability:** Why we need `extern "C"` interfaces.
3.  **Factory Export:** How the plugin exposes its capabilities.

## Practical Motivation
You want to ship a "Core" application and let 3rd parties write "Filters" for it without giving them your source code or recompiling the core.

## Step-by-Step Instructions
1.  Define a header-only interface `IPlugin`.
2.  Write a separate CMake project for a plugin (produces `.dll` or `.so`).
3.  Write the Host application that loads the DLL and calls a factory function to get the plugin instance.

## Verification
*   The Host app should run, find the DLL, load it, and execute a function from it.
