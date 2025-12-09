
# `torchfbm`
### Differentiable Fractional Brownian Motion & Rough Volatility for PyTorch

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/Topic-Rough_Volatility-b31b1b.svg)](https://arxiv.org/abs/1410.3394)

**`torchfbm`** is a high-performance, GPU-accelerated library for generating and analyzing Fractional Brownian Motion (fBm) and Fractional Gaussian Noise (fGn).

Designed for **Quantitative Finance** (Rough Volatility, Geometric fBm) and **Deep Reinforcement Learning** (Regime-Aware Exploration, Neural SDEs), it provides differentiable generators and layers that seamlessly integrate into the PyTorch ecosystem.

---

## 🚀 Features

### **Core Math (GPU Optimized)**
*   **Fast Generation:** Davies–Harte algorithm (FFT-based) for $O(N \log N)$ complexity.
*   **Exact Generation:** Cholesky decomposition for $O(N^3)$ ground-truth validation.
*   **Differentiable Estimators:** Vectorized Hurst exponent estimation (`estimate_hurst`) for meta-learning.

### **Quantitative Finance**
*   **Rough Processes:** `fractional_ou_process` (Fractional Ornstein-Uhlenbeck) for volatility modeling.
*   **Asset Pricing:** `geometric_fbm` for simulating asset paths with long memory ($H>0.5$) or roughness ($H<0.5$).
*   **Constraints:** `reflected_fbm` and `fractional_brownian_bridge` for boundary-constrained modeling.

### **Deep Learning & RL**
*   **Noisy Layers:** `FBMNoisyLinear` for replacing standard weights with correlated noise.
*   **Positional Embeddings:** `FractionalPositionalEmbedding` for Transformers on fractal data.
*   **RL Exploration:** `FBMActionNoise` (Stable Baselines3 compatible) for "Pink Noise" exploration.

---

## 📦 Install

**From PyPI (Coming Soon):**
```bash
pip install torchfbm
```

**For Development (Editable):**
```bash
git clone https://github.com/yourusername/torchfbm.git
cd torchfbm
pip install -e .
```

---

## ⚡ Quick Usage

### 1. Generate Rough Paths
Generate fractional noise on CUDA using the fast Davies-Harte method.

```python
import torch
from torchfbm import fbm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generate 4 paths of length 1024 with H=0.7 (Trending/Smooth)
path = fbm(n=1024, H=0.7, size=(4,), method='davies_harte', device=device)
```

### 2. Deep Learning (Regime-Aware Layers)
Replace standard `nn.Linear` with `FBMNoisyLinear` to inject correlated noise into weights.

```python
from torchfbm import FBMNoisyLinear

# Initialize layer with H=0.5 (Standard)
layer = FBMNoisyLinear(32, 10, H=0.5, device=device)

# Forward pass (Training mode enables noise)
x = torch.randn(8, 32, device=device)
layer.train()
y = layer(x)

# Dynamic Regime Switching (e.g., during market crash)
layer.H = 0.2  # Switch to Rough/Anti-correlated noise
layer.refresh_noise_stream()
```

### 3. Financial Processes
Simulate Geometric fBm (Stock Prices) and Fractional OU (Volatility).

```python
from torchfbm import geometric_fbm, fractional_ou_process

# Stock Price Simulation (Geometric fBm)
s = geometric_fbm(n=1000, H=0.7, mu=0.05, sigma=0.2, s0=100.0, device=device)

# Volatility Simulation (Rough OU)
v = fractional_ou_process(n=2048, H=0.1, theta=0.2, mu=0.0, sigma=0.5, device=device)
```

### 4. RL Action Noise
Compatible with Stable Baselines3 for continuous control.

```python
from torchfbm import FBMActionNoise

# H=0.7 creates "Sticky" exploration (good for robotics/momentum)
noise = FBMActionNoise(mean=0.0, sigma=0.2, H=0.7, size=(1,), device=device)
action = noise()
```

---

## 📈 Analysis Tools

### Differentiable Hurst Estimation
Estimate the roughness of a time series in a differentiable manner.

```python
from torchfbm import estimate_hurst

# Input shape: (Batch, Time)
H_est = estimate_hurst(path.unsqueeze(0), min_lag=4, max_lag=64)
print(f"Estimated H: {H_est.item():.4f}")
```

---

## 📝 Notes

*   **Speed vs Accuracy:** Use `method='davies_harte'` for large simulations ($N > 2000$). Use `method='cholesky'` for mathematical verification.
*   **Stability:** The Hurst parameter $H$ is clamped to $[0.01, 0.99]$ to avoid singularities at $H=0$ (Pink Noise) and $H=1$ (Ballistic).
*   **License:** MIT License. Free for research and commercial use.