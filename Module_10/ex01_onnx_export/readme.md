# Exercise 01: ONNX Export

## Goal
Export a simple PyTorch neural network model to the Open Neural Network Exchange (ONNX) format. This is the first step in deploying deep learning models to C++.

## Learning Objectives
1.  Define a simple neural network in PyTorch.
2.  Understand the `torch.onnx.export` function.
3.  Visualize the exported model using Netron (optional).

## Practical Motivation
PyTorch is great for training, but for production deployment in C++, we often use inference engines like ONNX Runtime or TensorRT. ONNX serves as the bridge format.

**Analogy:** Think of PyTorch as the "Microsoft Word" document where you write and edit your book (train your model). It has lots of editing tools. ONNX is like the "PDF" file you export to. It's meant for distribution and reading (inference) on any device, but you don't edit it anymore.

## Theory: ONNX
ONNX (Open Neural Network Exchange) is an open format built to represent machine learning models. It defines a common set of operators - the building blocks of machine learning and deep learning models - and a common file format to enable AI developers to use models with a variety of frameworks, tools, runtimes, and compilers.

## Step-by-Step Instructions

### Task 1: Install Dependencies
Ensure you have `torch` and `onnx` installed in your Python environment.
```bash
pip install torch onnx
```

### Task 2: Define a Simple Model
Open `todo/src/export_model.py`.
1.  Define a class `SimpleModel` inheriting from `torch.nn.Module`.
2.  In `__init__`, define a fully connected layer (Linear) and an activation (ReLU).
3.  In `forward`, define the data flow.

### Task 3: Export to ONNX
1.  Create an instance of the model.
2.  Create a dummy input tensor (e.g., `torch.randn(1, input_size)`).
3.  Call `torch.onnx.export(...)`.
    *   Specify the model, dummy input, and output filename (e.g., `simple_model.onnx`).
    *   Set `input_names` and `output_names`.
    *   Set `opset_version` (e.g., 11 or 13).

## Verification
Run the python script. It should generate `simple_model.onnx`.
```bash
python src/export_model.py
```
Check if the file exists.
