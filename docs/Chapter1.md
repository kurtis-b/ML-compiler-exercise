# Chapter 1: Introduction and project setup
Credits: The blog article by [Jeremy Kun](https://www.jeremykun.com/2023/08/10/mlir-getting-started/) helped me a lot in building this pipeline, especially this [one](https://www.jeremykun.com/2023/11/01/mlir-lowering-through-llvm/) on lowering MLIR dialects.
For a general introduction, I highly recommend Stephan Diehl’s [MLIR GPU lowering posts](https://www.stephendiehl.com/tags/mlir/).

## Introduction
MLIR is increasingly used in the compilation of machine learning models. Unlike traditional compilers, IREE’s compilation pipeline consists of multiple phases in which IR can be represented in different MLIR dialects, each with its own instruction set. A classification of these dialects can be found [here](https://youtu.be/hIt6J1_E21c?t=800). This multi-phase approach provides multiple abstraction levels, enabling better device-targeting opportunities—something that is significantly harder to achieve with a traditional two-phase compiler. Different optimizations can be applied at the specific abstraction level where they are most effective.

Typically, you start with a program that imports models from their defining framework (e.g., PyTorch, TensorFlow, ONNX) into a high-level MLIR dialect such as the linalg dialect. For example, torch-mlir reads a PyTorch model’s graph representation and converts it into the torch dialect (or directly into linalg). From there, the power of MLIR comes into play: many passes exist to incrementally lower dialects from high-level ones to more assembly-like dialects (the so-called exit dialects, e.g., llvm). Along the way, many optimizations can be applied at the appropriate dialect level.

MLIR also allows you to define your own dialects, passes, and transformations. Different types of passes exist: transform passes that modify code within a dialect, conversion passes that translate code between dialects, and analysis passes such as data-flow analyses.

Once the IR reaches an exit dialect, it can be translated to LLVM IR, leaving the MLIR domain. From there, object code can be generated for a specific hardware target, and the resulting model can be called from C++ like a regular function.

## What This Tutorial Is About
In this tutorial, we will build our own MLIR pipeline to lower real ML models. Specifically, we will take models from PyTorch, import them using [torch-mlir](https://github.com/llvm/torch-mlir), and apply a sequence of passes to lower the model to the MLIR LLVM dialect. We will then export the model to x86 assembly and call it from C/C++.

