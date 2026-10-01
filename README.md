PolyReLU: Learnable Piecewise Polynomial Activation FunctionPolyReLU: Learnable Piecewise Polynomial Activation Functions



PolyReLU is a novel, self-optimizing activation function designed for modern transformer architectures. By introducing learnable parameters for slope scaling (a) and leakage control (b) into piecewise polynomials (C1 Cubic and C2 Quintic variants), PolyReLU allows neural network layers to shape their own non-linearities via backpropagation autonomously.



Key Features



Dynamic Layer-Wise Adaptation: Rather than forcing a static activation across the entire network, each layer independently tunes its parameters during training.



High-Order Smoothness: Available in Cubic (C1) and Quintic (C2) smooth variants to ensure continuous gradient propagation without hard vertex cutoffs.



Optimized Execution: Features a custom index-masked architecture designed to bypass heavy runtime latencies and avoid cross-platform JIT compilation bottlenecks.



Empirical Benchmarks (NanoGPT FFN Blocks)

In head-to-head tournament mode against standard static activation functions inside NanoGPT feed-forward blocks, PolyReLU demonstrates superior performance:



Activation Function | Peak Test Accuracy | Final Cross-Entropy Loss

GELU (Baseline) | 97.40% | \~0.045

SiLU / Swish | 95.60% - 97.56% | \~0.048

Cubic PolyReLU (C1) | 97.66% | \~0.039

Quintic PolyReLU (C2) | 97.74% | \~0.040



Emergent Parameter Dynamics

Auditing layer-wise parameter evolution reveals an autonomous specialization:



Early Layers (Feature Extraction): The network pushes parameter a up (\~0.71-0.90) and drives leakage b down to 0.000000, locking in sharp, precise feature boundaries.



Deep Layers (Gradient Protection): The network dynamically re-introduces controlled leakage (b approx 0.025 - 0.041), preserving smooth gradient flow right before output projections.



License

This project is open-source under the terms specified in the LICENSE file.

