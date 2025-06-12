# Basic ML Tutorials

This repository offers a hands-on tutorial series on foundational machine learning concepts, designed to accompany the Week 2 lectures of the [REU'25 program](https://reu-ai-edge-osu.github.io/lectures.html) at [AI-EDGE](https://aiedge.osu.edu/).

Additional notes:
- [Environment Setup Guide](./notes/env.md)
- [Codebase Structure Overview](./notes/codebase.md)
- [Colab Usage Guide](./notes/colab.md)

## Table of Contents

| Module | Title                                       | Subsection                                              | Requires GPU? |
| ------ | ------------------------------------------- | ------------------------------------------------------- | ------------- |
| 1      | Online Perceptron for Linear Classification | 1.1: A toy example from slide 8                         | ❌ CPU-only    |
|        |                                             | 1.2: Perceptron on large-margin linearly separable data | ❌ CPU-only    |
|        |                                             | 1.3: Perceptron on small-margin linearly separable data | ❌ CPU-only    |
|        |                                             | 1.4: Perceptron on non-linearly separable data          | ❌ CPU-only    |
| 2      | From Taylor Expansions to Gradient Descent  | 2.1: Taylor approximation on toy functions              | ❌ CPU-only    |
|        |                                             | 2.2: Full-batch Gradient Descent                        | ❌ CPU-only    |
|        |                                             | 2.3: Compare stochastic vs full-batch Gradient Descent  | ❌ CPU-only    |
| 3      | Transformer for Binary Classification       | 3.1: Sequence classification using a Transformer encoder| ✅ CPU / GPU   |
| 4      | Transformer for Image Classification        | 4.1: Vision Transformer (ViT) on image patches          | ✅ GPU         |


## References
- [ziyueluocs/torch-tutorial](https://github.com/ziyueluocs/torch-tutorial) for environment setup
- [alochaus/taylor-series](https://github.com/alochaus/taylor-series) for sec 2.1
- [tintn/vision-transformer-from-scratch](https://github.com/tintn/vision-transformer-from-scratch) for sec 4.1
