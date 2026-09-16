# Custom Training Loop from Scratch with TensorFlow & `tf.GradientTape`

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An explicit, step-by-step implementation of an end-to-end training pipeline in TensorFlow from scratch, bypassing the standard `model.fit()` abstraction. This project demonstrates granular control over the forward pass, automatic differentiation, gradient computation, and parameter optimization using `tf.GradientTape`.

---

## 🎯 Overview & Motivation

While high-level APIs like `model.fit()` provide ease of use for standard models, real-world deep learning research and advanced production setups often require low-level control:
* **Custom Backpropagation:** Fine-grained monitoring and manipulation of gradients (e.g., gradient clipping, adversarial training, multi-task losses).
* **Non-standard Architectures:** Implementing complex training dynamics such as Generative Adversarial Networks (GANs) and Reinforcement Learning actor-critic loops.
* **Granular Debugging:** Direct visibility into numerical stability, vanishing/exploding gradients, and per-layer behavior.

---

## ⚙️ Technical Highlights

- **Data Pipeline (`tf.data`):** Optimized data loading with batching, shuffling, and normalization pipelines.
- **Model Architecture:** Modular Convolutional Neural Network (CNN) constructed via Keras Functional API.
- **Numerical Stability:** Unscaled logit outputs paired with `SparseCategoricalCrossentropy(from_logits=True)` to prevent numerical overflow/underflow.
- **Low-Level Training Step:**
  1. **Forward Pass:** Compute predictions within a `tf.GradientTape` context.
  2. **Loss Computation:** Measure discrepancy using sparse categorical cross-entropy.
  3. **Backward Pass:** Calculate exact gradients using reverse-mode automatic differentiation (`tape.gradient`).
  4. **Weight Updates:** Apply updates directly to `model.trainable_variables` via Adam optimizer (`apply_gradients`).
- **Evaluation Loop:** Independent evaluation phase measuring test set generalization using running metrics (`SparseCategoricalAccuracy`).

---

## 🧠 Architecture Overview

| Layer | Type | Specifications |
| :--- | :--- | :--- |
| **Input** | `Input` | `(28, 28, 1)` |
| **Feature Extraction** | `Conv2D` + `ReLU` | 32 filters, (3, 3) kernel |
| **Downsampling** | `MaxPooling2D` | (2, 2) pool size |
| **Reshape** | `Flatten` | Vectorized features |
| **Dense** | `Dense` + `ReLU` | 128 units |
| **Output (Logits)** | `Dense` (Linear) | 10 classes (digits 0–9) |

---

## 📊 Results

Trained on the MNIST benchmark dataset:

| Metric | Value |
| :--- | :--- |
| **Initial Loss** | `~2.30` (Consistent with uniform random initialization over 10 classes: $-\ln(0.1)$) |
| **Convergence** | Steady loss reduction per mini-batch update |
| **Test Accuracy** | **~89.6%** (Achieved within rapid exploratory training epochs) |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow numpy matplotlib
