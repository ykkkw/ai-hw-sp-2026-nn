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

**Parameters**: 235,146

**Why this structure**: MLP is the simplest possible baseline — no spatial assumptions, just raw pixel values fed through dense layers. Two hidden layers (256 → 128) give enough capacity to learn digit patterns while keeping the model lightweight. Dropout(0.3) prevents overfitting on the relatively small dataset.

---

### 2. CNN — Convolutional Neural Network
Three convolutional blocks to extract spatial features, followed by a fully-connected head.

| Block | Detail |
|-------|--------|
| Conv Block 1 | Conv2d(1→32, k=3), Conv2d(32→64, k=3), MaxPool2d(2×2), Dropout(0.25) |
| Conv Block 2 | Conv2d(64→128, k=3), MaxPool2d(2×2), Dropout(0.25) |
| FC Head | Linear(3200→256), ReLU, Dropout(0.5), Linear(256→10) |

**Parameters**: 914,698

**Why this structure**: CNNs exploit the spatial structure of images through local receptive fields and weight sharing, making them a natural fit for pixel grids. Stacking two conv blocks with progressive channel widening (1→32→64→128) lets the network learn low-level edges first, then higher-level shapes. MaxPooling after each block reduces spatial dimensions while retaining the most salient features, and heavier Dropout(0.5) on the FC head guards against overfitting.

---

### 3. Transformer Encoder
Vision-Transformer-style encoder. Splits each image into horizontal patches, projects them into embeddings, prepends a CLS token, and classifies from the CLS output.

| Component | Detail |
|-----------|--------|
| Patch Embedding | 7 patches of height 4, Linear(112 → 128) |
| CLS Token | Learnable parameter |
| Transformer | 3 encoder layers, d_model=128, 4 heads, FFN dim=256, Dropout(0.1) |
| Head | Linear(128 → 10) |

**Parameters**: 413,322

**Why this structure**: The Transformer treats each horizontal strip of the image as a token, allowing self-attention to capture long-range relationships across the full image — something neither MLP nor CNN can do directly. A small d_model=128 with 4 heads keeps the model compact for MNIST's modest scale. The CLS token approach (borrowed from BERT) aggregates global context from all patches into a single vector for classification.

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

## Results

| Model | Params | Test Accuracy | Best Val Epoch |
|-------|--------|--------------|----------------|
| MLP | 235K | **98.14%** | 11 |
| CNN | 915K | **99.45%** | 15 |
| Transformer Encoder | 413K | **97.00%** | 15 |

### Training Curves

<img width="1861" height="727" alt="training_curves_dark" src="https://github.com/user-attachments/assets/3a0f8def-ad7a-4be7-acdc-b4782e90cc27" />


### Key Observations
- **CNN wins** — spatial inductive bias (convolutions + pooling) is a natural fit for image data. Val accuracy hits 98.3% already at epoch 1.
- **MLP performs surprisingly well** — 98.1% with only ~235K params and no spatial priors.
- **Transformer is competitive but needs more** — 97% is solid, but lags because it lacks inductive bias and MNIST's 60K samples is modest for attention-based models. More epochs or data augmentation would likely close the gap.

---

## Per-class Accuracy

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
