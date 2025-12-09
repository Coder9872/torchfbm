import torch
import torch.nn as nn
import numpy as np
import time
import sys
import os
from pathlib import Path

# Ensure we can import the local package
# Add the repository root (parent of this file's parent) to sys.path so the
# local `torchfbm` package is discoverable regardless of current working dir.
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

try:
    from torchfbm import generators, processes, layers, sde, analysis, transforms, rl, loss, augmentations, estimators
    print(" Successfully imported torchfbm modules.")
except ImportError as e:
    print(f" ImportError: {e}")
    print("Make sure you run this script from the parent directory of 'torchfbm'.")
    sys.exit(1)

# Utility for printing
def log_test(name, result, msg=""):
    if result:
        print(f" [PASS] {name}: {msg}")
    else:
        print(f" [FAIL] {name}: {msg}")
        sys.exit(1)

def run_all_tests():
    print("\n" + "="*50)
    print("STARTING TORCH-FBM FULL SUITE TEST")
    print("="*50 + "\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on device: {device}\n")

    # ==========================================
    # 1. GENERATORS (Core Math)
    # ==========================================
    print("--- Testing Generators ---")
    
    # Test Davies-Harte
    fgn = generators.generate_davies_harte(n=1000, H=0.7, size=(4,), device=device)
    log_test("Davies-Harte Generation", fgn.shape == (4, 1000), "Shape correct")

    # Test Cholesky (Fallback)
    fgn_chol = generators.generate_cholesky(n=100, H=0.7, size=(2,), device=device)
    log_test("Cholesky Generation", fgn_chol.shape == (2, 100), "Shape correct")
    
    # Test fbm() wrapper
    path = generators.fbm(n=100, H=0.5, size=(1,), method='davies_harte', device=device)
    log_test("fBm Wrapper", path.shape == (1, 101), "Path has N+1 points (starts at 0)")

    # ==========================================
    # 2. PROCESSES (Financial/Physics)
    # ==========================================
    print("\n--- Testing Processes ---")

    # Geometric fBm
    s0 = 100.0
    gfbm = processes.geometric_fbm(n=100, H=0.4, s0=s0, device=device)
    is_positive = bool((gfbm > 0).all().item())
    log_test("Geometric fBm", is_positive, "Prices are strictly positive")

    # Reflected fBm
    lower, upper = -2.0, 2.0
    rfbm = processes.reflected_fbm(n=500, H=0.6, lower=lower, upper=upper, device=device)
    in_bounds = bool(((rfbm >= lower) & (rfbm <= upper)).all().item())
    log_test("Reflected fBm", in_bounds, f"Stays within [{lower}, {upper}]")

    # Fractional Brownian Bridge
    start, end = 10.0, 50.0
    bridge = processes.fractional_brownian_bridge(n=200, H=0.3, start_val=start, end_val=end, device=device)
    start_matches = torch.isclose(bridge[..., 0], torch.tensor(start, device=device)).all().item()
    end_matches = torch.isclose(bridge[..., -1], torch.tensor(end, device=device)).all().item()
    endpoints_match = bool(start_matches and end_matches)
    log_test("Brownian Bridge", endpoints_match, f"Starts at {start}, Ends at {end}")

    # ==========================================
    # 3. LAYERS & NN (Deep Learning)
    # ==========================================
    print("\n--- Testing Neural Layers ---")

    # FBM Noisy Linear
    layer = layers.FBMNoisyLinear(32, 10, H=0.8).to(device)
    x = torch.randn(4, 32).to(device)
    
    # Train mode (Noise active)
    layer.train()
    y1 = layer(x)
    y2 = layer(x) # Should be different because step advanced
    log_test("FBM Noisy Layer (Train)", not torch.allclose(y1, y2), "Output changes (Noise active)")
    
    # Eval mode (Noise off)
    layer.eval()
    y3 = layer(x)
    y4 = layer(x)
    log_test("FBM Noisy Layer (Eval)", torch.allclose(y3, y4), "Output deterministic (Noise off)")

    # Fractional Positional Embeddings
    emb = layers.FractionalPositionalEmbedding(max_len=50, d_model=16, H_range=(0.1, 0.9)).to(device)
    dummy_input = torch.zeros(2, 30, 16).to(device) # Batch 2, Seq 30, Dim 16
    out = emb(dummy_input)
    log_test("Fractional Pos Emb", out.shape == (2, 30, 16), "Output shape matches input")

    # Fractional Weight Init
    linear_layer = nn.Linear(100, 100).to(device)
    layers.fractional_init_(linear_layer.weight, H=0.8)
    log_test("Fractional Init", True, "Weights initialized in-place without error")

    # ==========================================
    # 4. ANALYSIS & TRANSFORMS (Quant Tools)
    # ==========================================
    print("\n--- Testing Analysis & Transforms ---")

    # Hurst Estimator (Differentiable)
    # Generate a long path with H=0.75
    true_H = 0.75
    test_path = generators.fbm(n=2000, H=true_H, size=(10,), device=device) # Batch of 10
    est_H = estimators.estimate_hurst(test_path)
    mean_est = est_H.mean().item()
    # Estimators are noisy, check if within reasonable range (0.65 - 0.85)
    valid_est = 0.65 <= mean_est <= 0.85
    log_test("Hurst Estimator", valid_est, f"True H={true_H}, Est H={mean_est:.3f}")

    # Fractional Diff
    ts = torch.randn(1, 100).to(device)
    diffed = transforms.fractional_diff(ts, d=0.4)
    log_test("FracDiff", diffed.shape == (1, 100), "Shape preserved after FracDiff")

    # ==========================================
    # 5. RL (Reinforcement Learning)
    # ==========================================
    print("\n--- Testing RL Components ---")
    
    action_noise = rl.FBMActionNoise(mean=0, sigma=0.1, H=0.3, size=(2,))
    n1 = action_noise()
    n2 = action_noise()
    # Use torch comparison instead of numpy to avoid compatibility issues
    arrays_different = not torch.allclose(n1, n2, atol=1e-6)
    log_test("RL Action Noise", n1.shape == (2,) and arrays_different, "Generates arrays for RL environments")

    # ==========================================
    # 6. NEURAL SDE (Advanced Research)
    # ==========================================
    print("\n--- Testing Neural fSDE (Differentiable H) ---")

    class SimpleDrift(nn.Module):
        def forward(self, x): return -0.5 * x # Mean reverting

    class SimpleDiff(nn.Module):
        def forward(self, x): return torch.ones_like(x) * 0.1

    # Learnable H
    model = sde.NeuralFSDE(state_size=1, drift_net=SimpleDrift(), diffusion_net=SimpleDiff(), 
                          H_init=0.5, learnable_H=True).to(device)
    
    # Forward Pass
    x0 = torch.zeros(4, 1).to(device) # Batch 4
    path = model(x0, n_steps=50)
    log_test("Neural fSDE Forward", path.shape == (4, 51, 1), "Path generated (Batch, Time, State)")

    # Backward Pass (The Holy Grail Check)
    loss_val = path.sum()
    loss_val.backward()
    grad_exists = model.raw_h.grad is not None
    log_test("Differentiable H", grad_exists, "Gradient flows to H parameter (Meta-Learning Ready)")

    # ==========================================
    # 7. LOSS FUNCTIONS
    # ==========================================
    print("\n--- Testing Loss Functions ---")
    
    # Hurst Regularization
    # We want output to have H=0.3
    reg_loss = loss.HurstRegularizationLoss(target_H=0.3)
    # We feed it the path generated above (which had H~0.5)
    l_val = reg_loss(path.squeeze(-1))
    log_test("Hurst Reg Loss", l_val.item() > 0, "Loss calculated successfully")
    
    # ==========================================
    # 8. AUGMENTATIONS
    # ==========================================
    print("\n--- Testing Augmentations ---")
    aug = augmentations.FractionalNoiseAugmentation(H=0.2, p=1.0).to(device)
    data = torch.zeros(10, 50).to(device)
    aug_data = aug(data)
    log_test("Frac Augmentation", not torch.allclose(data, aug_data), "Data modified by fGn noise")

    print("\n" + "="*50)
    print("ALL PASSED")
    print("="*50)

    return True

def run_mathematical_validation_tests():
    """Run additional mathematical property validation tests"""
    print("\n" + "="*60)
    print("MATHEMATICAL PROPERTIES VALIDATION")
    print("="*60 + "\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)  # For reproducible testing
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: H parameter boundary behavior
    print("--- Mathematical Test 1: H Parameter Boundaries ---")
    total_tests += 1
    try:
        for H in [0.01, 0.05, 0.5, 0.95, 0.99]:
            fgn = generators.generate_davies_harte(n=500, H=H, size=(10,), device=device)
            assert torch.isfinite(fgn).all(), f"Non-finite values at H={H}"
            # Test variance scaling (should be roughly constant across different H)
            var = torch.var(fgn)
            assert 0.01 < var < 100, f"Unrealistic variance {var:.3f} at H={H}"
        print("H boundary behavior: All H values produce finite, reasonable outputs")
        tests_passed += 1
    except Exception as e:
        print(f"H boundary test failed: {e}")
    
    # Test 2: Self-similarity property check
    print("\n--- Mathematical Test 2: Self-Similarity Property ---")
    total_tests += 1
    try:
        H = 0.7
        n = 1000
        scale = 2.0
        batch_size = 256  # Increased from 50 to reduce statistical variance
        
        # Original fBm
        fbm1 = generators.fbm(n=n, H=H, size=(batch_size,), device=device)
        
        # Scaled time fBm  
        fbm2 = generators.fbm(n=int(n*scale), H=H, size=(batch_size,), device=device)
        fbm2_sampled = fbm2[..., ::int(scale)] / (scale ** H)
        
        # Compare endpoint variances (should be similar due to self-similarity)
        var1 = torch.var(fbm1[..., -1], unbiased=False)
        var2 = torch.var(fbm2_sampled[..., -1], unbiased=False)
        var_ratio = (var1 / var2).item()
        
        # Allow 30% variance due to finite sampling
        assert 0.7 < var_ratio < 1.3, f"Self-similarity violated: ratio {var_ratio:.3f}"
        print(f"Self-similarity: Variance scaling ratio = {var_ratio:.3f}")
        tests_passed += 1
    except Exception as e:
        print(f" Self-similarity test failed: {e}")
    
    # Test 3: Bridge endpoint precision
    print("\n--- Mathematical Test 3: Bridge Endpoint Precision ---")
    total_tests += 1
    try:
        start_vals = [-5.2, 0.0, 3.7]
        end_vals = [1.1, -2.8, 10.5]
        
        for start_val, end_val in zip(start_vals, end_vals):
            bridge = processes.fractional_brownian_bridge(
                n=300, H=0.4, start_val=start_val, end_val=end_val, 
                size=(10,), device=device
            )
            start_error = torch.max(torch.abs(bridge[..., 0] - start_val)).item()
            end_error = torch.max(torch.abs(bridge[..., -1] - end_val)).item()
            
            assert start_error < 1e-5, f"Bridge start error {start_error:.2e} too large"
            assert end_error < 1e-5, f"Bridge end error {end_error:.2e} too large"
        
        print("Bridge endpoint precision: All constraints satisfied to machine precision")
        tests_passed += 1
    except Exception as e:
        print(f"Bridge endpoint test failed: {e}")
    
    # Test 4: Long-range dependence vs anti-persistence
    print("\n--- Mathematical Test 4: Long-Range Dependence ---")
    total_tests += 1
    try:
        n = 3000
        batch_size = 100  # Increased from 20 for better statistical stability
        
        # Persistent case (H > 0.5)
        H_persistent = 0.8
        fbm_persistent = generators.fbm(n=n, H=H_persistent, size=(batch_size,), device=device)
        increments_p = fbm_persistent[..., 1:] - fbm_persistent[..., :-1]
        
        # Anti-persistent case (H < 0.5)
        H_anti = 0.2
        fbm_anti = generators.fbm(n=n, H=H_anti, size=(batch_size,), device=device)
        increments_a = fbm_anti[..., 1:] - fbm_anti[..., :-1]
        
        # Calculate lag-1 autocorrelation
        def lag1_autocorr(increments):
            x, y = increments[..., :-1], increments[..., 1:]
            mean_x, mean_y = torch.mean(x), torch.mean(y)
            cov = torch.mean((x - mean_x) * (y - mean_y))
            std_x, std_y = torch.std(x), torch.std(y)
            return cov / (std_x * std_y)
        
        corr_p = lag1_autocorr(increments_p).item()
        corr_a = lag1_autocorr(increments_a).item()
        
        # H > 0.5 should give positive correlation, H < 0.5 should give negative
        assert corr_p > 0.05, f"Expected positive correlation for H={H_persistent}, got {corr_p:.4f}"
        assert corr_a < -0.05, f"Expected negative correlation for H={H_anti}, got {corr_a:.4f}"
        
        print(f"Long-range dependence: H={H_persistent} → r={corr_p:.3f}, H={H_anti} → r={corr_a:.3f}")
        tests_passed += 1
    except Exception as e:
        print(f"Long-range dependence test failed: {e}")
    
    # Test 5: Generator algorithm consistency
    print("\n--- Mathematical Test 5: Algorithm Consistency ---")
    total_tests += 1
    try:
        H = 0.6
        n = 500
        batch_size = 100  # Increased from 20 for better statistical stability
        
        # Generate with both methods using same parameters
        torch.manual_seed(111)
        fgn_dh = generators.generate_davies_harte(n=n, H=H, size=(batch_size,), device=device)
        torch.manual_seed(111) 
        fgn_chol = generators.generate_cholesky(n=n, H=H, size=(batch_size,), device=device)
        
        # Compare statistical properties (not exact values due to different algorithms)
        var_dh = torch.var(fgn_dh)
        var_chol = torch.var(fgn_chol)
        mean_dh = torch.mean(fgn_dh)
        mean_chol = torch.mean(fgn_chol)
        
        var_ratio = (var_dh / var_chol).item()
        mean_diff = torch.abs(mean_dh - mean_chol).item()
        
        # Both should have similar variance (within factor of 2) and near-zero mean
        assert 0.5 < var_ratio < 2.0, f"Variance ratio DH/Chol = {var_ratio:.3f} out of range"
        assert mean_diff < 0.1, f"Mean difference {mean_diff:.3f} too large"
        
        print(f"Algorithm consistency: Variance ratio = {var_ratio:.3f}, Mean diff = {mean_diff:.4f}")
        tests_passed += 1
    except Exception as e:
        print(f"Algorithm consistency test failed: {e}")
    
    print(f"\n" + "="*60)
    print(f" MATHEMATICAL VALIDATION: {tests_passed}/{total_tests} TESTS PASSED")
    if tests_passed == total_tests:
        print(" All mathematical properties verified!")
        return True
    else:
        print(f"  {total_tests - tests_passed} mathematical tests failed")
        return False

def run_all_tests_extended():
    """Run both standard and mathematical tests"""
    standard_success = run_all_tests()
    if not standard_success:
        return False
        
    mathematical_success = run_mathematical_validation_tests()
    return mathematical_success

if __name__ == "__main__":
    success = run_all_tests_extended()
    sys.exit(0 if success else 1)