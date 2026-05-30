# HW4: MNIST Image Recognition — Three Neural Networks

**Dataset**: MNIST (60,000 train / 10,000 test, 28×28 grayscale handwritten digits)  
**Task**: Train three architectures from scratch; evaluate on the held-out test set.

---

## Models

### 1. MLP — Multi-Layer Perceptron
A shallow fully-connected network. Flattens each image into a 784-dim vector and passes it through two hidden layers.

| Layer | Detail |
|-------|--------|
| Input | 784 (28×28 flattened) |
| Hidden 1 | Linear(784 → 256), ReLU, Dropout(0.3) |
| Hidden 2 | Linear(256 → 128), ReLU, Dropout(0.3) |
| Output | Linear(128 → 10) |

**Why this structure**: MLP is the simplest possible baseline — no spatial assumptions, just raw pixel values fed through dense layers. Two hidden layers (256 → 128) give enough capacity to learn digit patterns while keeping the model lightweight. Dropout(0.3) prevents overfitting on the relatively small dataset. 

params = (784 × 256 + 256) + (256 × 128 + 128) + (128 * 10 + 10) = 235,146

---

### 2. CNN — Convolutional Neural Network
Three convolutional blocks to extract spatial features, followed by a fully-connected head.

| Block | Detail |
|-------|--------|
| Conv Block 1 | Conv2d(1→32, k=3), Conv2d(32→64, k=3), MaxPool2d(2×2), Dropout(0.25) |
| Conv Block 2 | Conv2d(64→128, k=3), MaxPool2d(2×2), Dropout(0.25) |
| FC Head | Linear(3200→256), ReLU, Dropout(0.5), Linear(256→10) |

**Why this structure**: CNNs exploit the spatial structure of images through local receptive fields and weight sharing, making them a natural fit for pixel grids. Stacking two conv blocks with progressive channel widening (1→32→64→128) lets the network learn low-level edges first, then higher-level shapes regardless of position. MaxPooling reduces spatial dimensions while retaining the most important features, and heavier Dropout(0.5) prevents overfitting.

params = (1 * 32 * 3 * 3 + 32) + (32 * 64 * 3 * 3 + 64) + (64 * 128 * 3 * 3 + 128) + (3200 * 256 + 256) + (256 * 10 + 10) = 914,698

---

### 3. Transformer Encoder
Vision-Transformer-style encoder. Splits each image into horizontal patches, projects them into embeddings, prepends a CLS token, and classifies from the CLS output.

| Component | Detail |
|-----------|--------|
| Patch Embedding | 7 patches of height 4, Linear(112 → 128) |
| CLS Token | Learnable parameter |
| Transformer | 3 encoder layers, d_model=128, 4 heads, FFN dim=256, Dropout(0.1) |
| Head | Linear(128 → 10) |

**Why this structure**: The Transformer treats each horizontal strip of the image as a token, allowing self-attention to capture relationships across the full image where neither MLP nor CNN can do directly.

params = (112 * 128 + 128) + 128 + 3 * [[4 * (128 * 128 + 128)] + [(128 * 256 + 256) + (256 * 128 + 128)] + (256 * 2)] + (128 * 10 + 10) = 413,322

---

## Training Setup

| Setting | Value |
|---------|-------|
| Optimizer | Adam, lr=0.001 |
| Loss | CrossEntropyLoss |
| Epochs | 15 |
| Batch size | 64 |
| Train/Val split | 80% / 20% (from 60k train set) |
| Model selection | Best validation accuracy checkpoint |
| Test set | Official MNIST test set (10,000 images) — never touched during training |

---
## Training Curves
<img width="1861" height="727" alt="training_curves_dark" src="https://github.com/user-attachments/assets/3a0f8def-ad7a-4be7-acdc-b4782e90cc27" />

## Accuracy Results

| Model | Params | Test Accuracy | Best Val Epoch |
|-------|--------|--------------|----------------|
| MLP | 235K | **98.14%** | 11 |
| CNN | 915K | **99.45%** | 15 |
| Transformer Encoder | 413K | **97.00%** | 15 |

### Key Observations
- **CNN wins** — spatial inductive bias (convolutions + pooling) is a natural fit for image data. Val accuracy hits 98.3% already at epoch 1.
- **MLP performs well** — 98.1% with only ~235K params and no spatial priors.
- **Transformer is competitive but needs more training** — 97% is solid, but lags because it lacks inductive bias and MNIST's 60K samples is modest for attention-based models. More epochs or data augmentation would likely close the gap.

### Per-class Accuracy

<img width="1710" height="713" alt="per_class_accuracy_dark" src="https://github.com/user-attachments/assets/814f6a84-c68f-4113-a002-4b29ad7808a6" />

- Digit **9** is the hardest for both MLP and CNN (often confused with 4 or 7).
- The Transformer struggles more broadly on digits **4, 5, 7, 8** — structurally ambiguous digits that benefit from the fine-grained spatial priors CNNs provide.

---

## Files

```
hw4/
├── model.py            # MLP, CNN, Encoder class definitions
├── train.py            # Training loop with validation & checkpointing
├── test.py             # Test evaluation with per-class accuracy
├── mlp_best.pt         # Saved MLP weights
├── cnn_best.pt         # Saved CNN weights
├── encoder_best.pt     # Saved Encoder weights
├── mlp_results.txt     # MLP test output
├── cnn_results.txt     # CNN test output
├── encoder_results.txt # Encoder test output
└── training_loss.txt   # Training logs for all three models
```

# HW7: Adversarial Attacks on MNIST CNN

**Goal**: Attack a pre-trained CNN (from HW4) using three adversarial attack algorithms and measure how well each fools the model.  
**Target model**: CNN trained on MNIST — 99.45% test accuracy (914K params)  
**Metric**: Recognition Rate (clean) + Attack Success Rate (ASR)

---

## Attack Algorithms

### 1. FGSM — Fast Gradient Sign Method
A single-step attack. Computes the gradient of the loss w.r.t. the input, then shifts every pixel one step in the direction that maximally increases the loss.

```
x_adv = x + ε · sign(∇ₓ J(x, y))
```

**Advantage**: Fast — one forward + one backward pass per batch. Good baseline for measuring model robustness. The simplicity also means it's the weakest of the three.

---

### 2. PGD — Projected Gradient Descent (Iterative FGSM)
Runs FGSM repeatedly for `K` iterations with a smaller step size `α`, projecting back into the ε-ball after each step. Starts from a random point within the ε-ball (random initialization) to avoid local optima.

```
x⁰   = x + uniform noise in [-ε, ε]
xᵏ⁺¹ = Proj_{x,ε}( xᵏ + α · sign(∇ₓ J(xᵏ, y)) )
```

**Advantage**: Considered the strongest first-order attack. Iterating with projection ensures the adversarial example stays within a perceptually small ε-ball around the original image. More iterations = stronger attack at the cost of compute.

---

### 3. Momentum I-FGSM — Momentum Iterative FGSM
Adds a momentum accumulator to I-FGSM. Instead of using the raw gradient each step, it maintains a running exponential average of past gradients (normalized by their L1 mean), which stabilizes the update direction across iterations.

```
g⁰   = 0
gᵏ⁺¹ = μ · gᵏ + ∇ₓ J(xᵏ, y) / ‖∇ₓ J(xᵏ, y)‖₁
xᵏ⁺¹ = Proj_{x,ε}( xᵏ + α · sign(gᵏ⁺¹) )
```

**Advantage**: Smoothed gradient path, and is the best for cross-model transfer.

---

## Setup

| Setting | Value |
|---------|-------|
| Target model | CNN (HW4 best checkpoint) |
| Epsilon (ε) | 0.15 |
| PGD / MI-FGSM step size (α) | 0.15/20 ≈ 0.0078 |
| PGD / MI-FGSM iterations | 20 |
| MI-FGSM decay (μ) | 1.0 |
| Test set | MNIST official test set (10,000 images) |
| ASR denominator | Correctly classified samples only |

---

## Results

| Metric | Value |
|--------|-------|
| Clean recognition rate | **98.20%** |
| FGSM ASR | **61.35%** |
| PGD ASR | **83.67%** |
| Momentum I-FGSM ASR | **81.76%** |

### Attack Success Rate

<img width="1072" height="521" alt="hw7_asr_dark" src="https://github.com/user-attachments/assets/9e8c0736-6438-4fb7-9056-ccd59b42e539" />

### Key Observations
- **PGD is the strongest attack** — 83.67% ASR with 20 iterations and random initialization thoroughly explores the ε-ball.
- **MI-FGSM is close behind** at 81.76% — the momentum term stabilizes gradient direction, but with iters=20 and the same α, it converges similarly to PGD on MNIST.
- **FGSM is the weakest** — a single step at ε=0.15 still fools 61% of correctly-classified samples, showing that even minimal perturbation can be highly effective on an undefended model.

---

## Files

```
hw7/
├── attack_functions.py   # FGSM, PGD, Momentum I-FGSM implementations
├── run_attacks.py        # Loads CNN, runs all three attacks, prints metrics
└── metrics_ASR.txt       # Output log with recognition rate and ASR results
```
