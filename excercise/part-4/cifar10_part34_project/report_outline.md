# Report Outline for Part 3.4 — CIFAR-10

## 1. Introduction

This section investigates different image embedding and hybrid architecture strategies for image classification on CIFAR-10. The focus is not only on classification performance, but also on how different tokenization and feature extraction designs influence Transformer-based models.

## 2. Dataset

- Dataset: CIFAR-10
- Number of classes: 10
- Image size: 32×32 RGB
- Reason for choosing CIFAR-10:
  - standard benchmark
  - small image size suitable for fast experimentation
  - appropriate for comparing multiple embedding strategies

## 3. Model Design

### 3.1 ViT with Standard Patch Embedding

- non-overlapping patches
- linear/conv patch projection
- positional embedding + Transformer encoder + classification head

### 3.2 ViT with Overlapping Patch Embedding

- overlapping patches using conv tokenizer
- same Transformer backbone for fairer comparison
- objective: analyze the effect of tokenization strategy

### 3.3 CNN + Transformer Hybrid

- CNN as local feature extractor
- feature map reshaped into tokens
- Transformer encoder captures long-range interaction

## 4. Experimental Setup

- optimizer
- learning rate
- batch size
- epochs
- train/validation/test split
- augmentation methods
- hardware

## 5. Results

Include:

- table of best validation accuracy, test accuracy, parameter count, and training time
- training/validation loss curves
- training/validation accuracy curves
- short summary of observations

## 6. Discussion

Suggested discussion points:

- whether overlap patch embedding improves token continuity
- whether CNN + Transformer works better on small images because CNN captures local patterns effectively
- trade-off between accuracy and computational cost
- effect of number of tokens on efficiency

## 7. Conclusion

Summarize which architecture works best for CIFAR-10 and explain why it is suitable under the scope of Part 3.4.
