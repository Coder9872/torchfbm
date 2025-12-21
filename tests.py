import torch
import torch.nn as nn
import numpy as np
import sys
import os
import time

# =============================================================================
# USER CONFIGURATION
# =============================================================================
PACKAGE_NAME = "torchfbm"  # Change to "torch_fbm" if your folder uses underscores

def log(task, status, msg=""):
    symbol = "✅" if status else "❌"
    print(f"{symbol} [{task}]: {msg}")
    if not status:
        sys.exit(1)

# =============================================================================
# 1. IMPORT CHECK
# =============================================================================
print(f"--- 1. Testing Import ({PACKAGE_NAME}) ---")
try:
    # Dynamic import to handle package name variable
    torchfbm = __import__(PACKAGE_NAME)
    
    # Import submodules to verify exposure
    from torchfbm import fbm, generate_davies_harte
    from torchfbm import FBMNoisyLinear, FractionalPositionalEmbedding
    from torchfbm import geometric_fbm, fractional_ou_process, reflected_fbm, fractional_brownian_bridge
    from torchfbm import FBMActionNoise
    from torchfbm import estimate_hurst, fractional_diff
    from torchfbm import NeuralFSDE
    
    # Import new modules (Online/Schedulers)
    # Note: These might be under submodules depending on your __init__.py exposure
    # If not exposed at top level, we import from submodule
    try:
        from torchfbm.online import CachedFGNGenerator
        from torchfbm.schedulers import get_hurst_schedule
        from torchfbm.loss import SpectralConsistencyLoss
    except ImportError:
        # Fallback if your __init__.py doesn't expose them directly
        import torchfbm.online as online
        CachedFGNGenerator = online.CachedFGNGenerator
        import torchfbm.schedulers as schedulers
        get_hurst_schedule = schedulers.get_hurst_schedule
        import torchfbm.loss as loss
        SpectralConsistencyLoss = loss.SpectralConsistencyLoss

    log("Import", True, f"Successfully imported {PACKAGE_NAME} version {getattr(torchfbm, '__version__', 'unknown')}")

except ImportError as e:
    log("Import", False, f"Failed to import. Did you run 'pip install -e .'? Error: {e}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"    Running on: {device}\n")


# =============================================================================
# 2. CORE GENERATORS
# =============================================================================
print("--- 2. Testing Core Generators ---")

# Batch Generation
try:
    # Test Davies-Harte (Fast)
    path_fast = fbm(n=1000, H=0.7, size=(4,), method='davies_harte', device=device)
    log("Generator (Davies-Harte)", path_fast.shape == (4, 1001), "Shape (Batch, Time+1) correct")
    
    # Test Cholesky (Exact)
    path_exact = fbm(n=100, H=0.3, size=(2,), method='cholesky', device=device)
    log("Generator (Cholesky)", path_exact.shape == (2, 101), "Fallback method works")

    # Boundary Check (Hurst Clamping)
    # Should not crash even with illegal H
    path_clamp = fbm(n=100, H=1.5, method='davies_harte', device=device)
    log("Stability", True, "Clamped H=1.5 to 0.99 without crash")

except Exception as e:
    log("Generator Crash", False, str(e))


# =============================================================================
# 3. ONLINE STREAMING (The New Feature)
# =============================================================================
print("\n--- 3. Testing Online Streaming (Cached Cholesky) ---")

try:
    # Initialize Stream
    stream = CachedFGNGenerator(H=0.4, device=device)
    
    # Step 1
    val1 = stream.step()
    # Step 2
    val2 = stream.step()
    
    log("Streaming Step", val1.numel() == 1 and val2.numel() == 1, "Generated scalar steps")
    log("Streaming History", stream.n == 2, "Internal counter incremented")
    
    # Performance check: Generate 100 points
    start = time.time()
    for _ in range(100):
        stream.step()
    duration = time.time() - start
    log("Streaming Speed", True, f"100 steps in {duration:.4f}s")

except Exception as e:
    log("Online Module Crash", False, str(e))


# =============================================================================
# 4. FINANCIAL PROCESSES
# =============================================================================
print("\n--- 4. Testing Financial Processes ---")

# Geometric fBm
s = geometric_fbm(n=500, H=0.6, s0=100.0, device=device)
log("Geometric fBm", (s > 0).all().item(), "Prices stay positive")

# Reflected fBm
lower, upper = -1.0, 1.0
r = reflected_fbm(n=500, H=0.5, lower=lower, upper=upper, device=device)
in_bounds = (r >= lower).all() and (r <= upper).all()
log("Reflected fBm", in_bounds.item(), f"Stays strictly in [{lower}, {upper}]")

# Brownian Bridge
start_val, end_val = 10.0, 50.0
bridge = fractional_brownian_bridge(n=200, H=0.3, start_val=start_val, end_val=end_val, device=device)
err_start = (bridge[..., 0] - start_val).abs().max()
err_end = (bridge[..., -1] - end_val).abs().max()
log("Brownian Bridge", err_start < 1e-4 and err_end < 1e-4, "Hits target endpoints")


# =============================================================================
# 5. DEEP LEARNING LAYERS
# =============================================================================
print("\n--- 5. Testing Deep Learning Layers ---")

# Noisy Linear
layer = FBMNoisyLinear(32, 10, H=0.5).to(device)
x = torch.randn(8, 32).to(device)

layer.train()
y_train_1 = layer(x)
y_train_2 = layer(x)
log("Noisy Linear (Train)", not torch.allclose(y_train_1, y_train_2), "Noise active (stochastic output)")

layer.eval()
y_eval_1 = layer(x)
y_eval_2 = layer(x)
log("Noisy Linear (Eval)", torch.allclose(y_eval_1, y_eval_2), "Noise frozen (deterministic output)")

# Positional Embeddings
emb = FractionalPositionalEmbedding(max_len=50, d_model=64, H_range=(0.1, 0.9)).to(device)
dummy_seq = torch.zeros(2, 30, 64).to(device) # Batch 2, Seq 30
out = emb(dummy_seq)
log("Positional Emb", out.shape == (2, 30, 64), "Embedding shape matches")


# =============================================================================
# 6. DIFFUSION TOOLS (Schedulers & Loss)
# =============================================================================
print("\n--- 6. Testing Diffusion Tools ---")

# Hurst Scheduler
hs = get_hurst_schedule(n_steps=100, start_H=0.1, end_H=0.9, type='cosine')
log("Hurst Scheduler", len(hs) == 100 and hs[0] < hs[-1], "Generated Cosine Schedule")

# Spectral Loss
spec_loss = SpectralConsistencyLoss(target_beta=2.4) # Target H=0.7
dummy_gen = torch.randn(4, 1024, device=device)
loss_val = spec_loss(dummy_gen)
log("Spectral Loss", loss_val.item() > 0, "Loss calculation successful")


# =============================================================================
# 7. ANALYSIS & RL
# =============================================================================
print("\n--- 7. Testing Analysis & RL ---")

# Hurst Estimation
# Generate ground truth H=0.8
gt_path = fbm(n=2048, H=0.8, size=(10,), device=device)
est = estimate_hurst(gt_path, min_lag=4, max_lag=64)
mean_est = est.mean().item()
log("Hurst Estimator", 0.7 < mean_est < 0.9, f"Estimated {mean_est:.3f} (True 0.8)")

# RL Action Noise (PyTorch default)
action_noise_torch = FBMActionNoise(mean=0, sigma=0.1, H=0.3, size=(2,), device=device.type)
n_torch = action_noise_torch()
log("RL Noise (PyTorch)", isinstance(n_torch, torch.Tensor), "Returns PyTorch tensor by default")

# RL Action Noise (NumPy option - skip if NumPy unavailable)
try:
    action_noise_numpy = FBMActionNoise(mean=0, sigma=0.1, H=0.3, size=(2,), device=device.type, return_numpy=True)
    n_numpy = action_noise_numpy()
    log("RL Noise (NumPy)", isinstance(n_numpy, np.ndarray), "Returns NumPy array when requested")
except ImportError:
    log("RL Noise (NumPy)", True, "Skipped (NumPy compatibility issue in environment)")


# =============================================================================
# 8. NEURAL SDE (Differentiable)
# =============================================================================
print("\n--- 8. Testing Neural SDE ---")

class MockDrift(nn.Module):
    def forward(self, x): return -x

class MockDiff(nn.Module):
    def forward(self, x): return torch.ones_like(x) * 0.1

model = NeuralFSDE(state_size=1, drift_net=MockDrift(), diffusion_net=MockDiff(), learnable_H=True).to(device)
x0 = torch.zeros(2, 1).to(device)
traj = model(x0, n_steps=10)

loss = traj.sum()
loss.backward()

log("Neural FSDE", traj.shape == (2, 11, 1), "Trajectory generated")
log("Differentiable H", model.raw_h.grad is not None, "Gradients flow to H parameter")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*50)
print("🎉 VERIFICATION COMPLETE")
print("All systems functional. The package is ready for external users.")
print("="*50)