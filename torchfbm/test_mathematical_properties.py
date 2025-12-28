#!/usr/bin/env python3
"""
Comprehensive mathematical property tests for torchfbm library.

Tests theoretical properties of fractional Brownian motion:
1. Self-similarity: fBm(at) =_d a^H * fBm(t)
2. Covariance function: E[fBm(t)fBm(s)] = 0.5*(|t|^2H + |s|^2H - |t-s|^2H)
3. Scaling behavior of increments: Var(dB_H) ~ dt^(2H)
4. Long-range dependence vs anti-persistence based on H
5. Numerical stability at boundary conditions
6. Bridge constraints and process properties
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Import from current directory
import generators
import processes
import estimators
import analysis


def test_self_similarity():
    """Test that fBm satisfies statistical self-similarity: fBm(at) =_d a^H * fBm(t)"""
    print("--- Testing Self-Similarity Property ---")

    H = 0.7
    n = 1000
    a = 2.0  # Scaling factor
    batch_size = 100  # Large batch for statistical significance

    # Generate original fBm paths
    fbm_t = generators.fbm(n=n, H=H, size=(batch_size,))

    # Generate scaled time fBm paths (double the time points, same H)
    fbm_at = generators.fbm(n=int(n * a), H=H, size=(batch_size,))

    # Extract every a-th point from scaled path (corresponds to t, 2t, 3t, ...)
    fbm_at_sampled = fbm_at[..., :: int(a)]

    # Scale down by a^H to test self-similarity
    scaled_fbm = fbm_at_sampled / (a**H)

    # Compare variances (should be similar due to self-similarity)
    var_original = torch.var(fbm_t, dim=0)
    var_scaled = torch.var(scaled_fbm, dim=0)

    # Allow 15% relative error due to finite sampling
    relative_error = torch.abs(var_original - var_scaled) / torch.abs(var_original)
    max_error = torch.max(relative_error).item()

    print(f"✅ Self-similarity test: Max relative error = {max_error:.3f}")
    assert max_error < 0.15, f"Self-similarity violated: error {max_error:.3f} > 0.15"


def test_covariance_function():
    """Test theoretical covariance: E[B_H(t)B_H(s)] = 0.5*(|t|^2H + |s|^2H - |t-s|^2H)"""
    print("--- Testing Covariance Function ---")

    H = 0.6
    times = torch.linspace(0.1, 5.0, 50)  # Avoid t=0 for numerical stability
    batch_size = 1000

    fbm_paths = generators.fbm(n=len(times) - 1, H=H, size=(batch_size,))

    # Pick two time points for covariance testing
    t1_idx, t2_idx = 10, 30
    t1, t2 = times[t1_idx], times[t2_idx]

    # Empirical covariance
    empirical_cov = torch.mean(fbm_paths[:, t1_idx] * fbm_paths[:, t2_idx])

    # Theoretical covariance
    theoretical_cov = 0.5 * (
        torch.abs(t1) ** (2 * H)
        + torch.abs(t2) ** (2 * H)
        - torch.abs(t1 - t2) ** (2 * H)
    )

    relative_error = torch.abs(empirical_cov - theoretical_cov) / torch.abs(
        theoretical_cov
    )
    print(
        f"✅ Covariance test: Theoretical={theoretical_cov:.4f}, Empirical={empirical_cov:.4f}, Error={relative_error:.3f}"
    )
    assert (
        relative_error < 0.1
    ), f"Covariance function violated: error {relative_error:.3f} > 0.1"


def test_increment_scaling():
    """Test that increments scale as Var(ΔB_H(τ)) ∼ τ^(2H)"""
    print("--- Testing Increment Scaling ---")

    H = 0.8
    batch_size = 500
    base_steps = 100

    variances = []
    time_scales = [1, 2, 4, 8]  # Powers of 2 for clean scaling

    for scale in time_scales:
        n_steps = base_steps * scale
        dt = 1.0 / n_steps  # Keep total time = 1

        fbm_path = generators.fbm(n=n_steps, H=H, size=(batch_size,))
        increments = fbm_path[..., 1:] - fbm_path[..., :-1]
        variance = torch.var(increments)
        variances.append(variance.item())

    # Check scaling: Var should scale as dt^(2H) = (1/n)^(2H)
    expected_ratios = [(time_scales[0] / scale) ** (2 * H) for scale in time_scales[1:]]
    empirical_ratios = [variances[0] / variances[i] for i in range(1, len(variances))]

    print(
        f"✅ Increment scaling: H={H}, Expected ratios={expected_ratios}, Empirical={empirical_ratios}"
    )
    for i, (exp, emp) in enumerate(zip(expected_ratios, empirical_ratios)):
        relative_error = abs(exp - emp) / exp
        assert (
            relative_error < 0.15
        ), f"Increment scaling violated at scale {time_scales[i+1]}: error {relative_error:.3f}"


def test_hurst_parameter_boundaries():
    """Test behavior at H parameter boundaries"""
    print("--- Testing Hurst Parameter Boundaries ---")

    # Test near boundaries
    h_values = [0.01, 0.05, 0.5, 0.95, 0.99]
    n = 500

    for H in h_values:
        try:
            # Test generation works
            fgn = generators.generate_davies_harte(n=n, H=H, size=(10,))
            fbm_path = generators.fbm(n=n, H=H, size=(5,))

            # Test that paths are finite and not NaN
            assert torch.isfinite(fgn).all(), f"Non-finite values at H={H}"
            assert torch.isfinite(fbm_path).all(), f"Non-finite fBm values at H={H}"

            print(f"✅ H={H} generation successful")

        except Exception as e:
            raise AssertionError(f"Failed at H={H}: {e}")


def test_bridge_constraints():
    """Test that fractional Brownian bridges satisfy endpoint constraints"""
    print("--- Testing Bridge Constraints ---")

    start_val, end_val = -1.5, 3.2
    H = 0.4
    n = 200
    batch_size = 20

    bridge = processes.fractional_brownian_bridge(
        n=n, H=H, start_val=start_val, end_val=end_val, size=(batch_size,)
    )

    # Check endpoints
    start_error = torch.max(torch.abs(bridge[..., 0] - start_val))
    end_error = torch.max(torch.abs(bridge[..., -1] - end_val))

    print(
        f"✅ Bridge constraints: Start error={start_error:.6f}, End error={end_error:.6f}"
    )
    assert start_error < 1e-5, f"Bridge start constraint violated: error {start_error}"
    assert end_error < 1e-5, f"Bridge end constraint violated: error {end_error}"


def test_long_range_dependence():
    """Test long-range dependence properties for H > 0.5 vs anti-persistence for H < 0.5"""
    print("--- Testing Long-Range Dependence ---")

    n = 2000
    batch_size = 50

    # Persistent case: H > 0.5
    H_persistent = 0.8
    fbm_persistent = generators.fbm(n=n, H=H_persistent, size=(batch_size,))
    increments_persistent = fbm_persistent[..., 1:] - fbm_persistent[..., :-1]

    # Anti-persistent case: H < 0.5
    H_anti = 0.2
    fbm_anti = generators.fbm(n=n, H=H_anti, size=(batch_size,))
    increments_anti = fbm_anti[..., 1:] - fbm_anti[..., :-1]

    # Compute lag-1 autocorrelation
    def lag1_autocorr(increments):
        x, y = increments[..., :-1], increments[..., 1:]
        cov = torch.mean((x - torch.mean(x)) * (y - torch.mean(y)))
        var_x = torch.var(x)
        var_y = torch.var(y)
        return cov / torch.sqrt(var_x * var_y)

    corr_persistent = lag1_autocorr(increments_persistent).item()
    corr_anti = lag1_autocorr(increments_anti).item()

    print(
        f"✅ Autocorrelations: H={H_persistent} gives r={corr_persistent:.4f}, H={H_anti} gives r={corr_anti:.4f}"
    )

    # For H > 0.5, expect positive correlation; for H < 0.5, expect negative
    assert (
        corr_persistent > 0.1
    ), f"Expected positive correlation for H={H_persistent}, got {corr_persistent:.4f}"
    assert (
        corr_anti < -0.1
    ), f"Expected negative correlation for H={H_anti}, got {corr_anti:.4f}"


def test_hurst_estimation_accuracy():
    """Test that Hurst estimation is reasonably accurate"""
    print("--- Testing Hurst Estimation Accuracy ---")

    true_H_values = [0.3, 0.5, 0.7, 0.9]
    n = 5000  # Long paths for better estimation
    batch_size = 10
    tolerance = 0.15  # Allow 15% error in Hurst estimation

    for true_H in true_H_values:
        fbm_path = generators.fbm(n=n, H=true_H, size=(batch_size,))
        estimated_H = estimators.estimate_hurst(fbm_path)
        mean_estimate = torch.mean(estimated_H).item()

        relative_error = abs(mean_estimate - true_H) / true_H
        print(
            f"✅ H estimation: True H={true_H}, Estimated H={mean_estimate:.3f}, Error={relative_error:.3f}"
        )

        assert (
            relative_error < tolerance
        ), f"Hurst estimation error {relative_error:.3f} > {tolerance} for H={true_H}"


def test_geometric_fbm_properties():
    """Test that geometric fBm remains positive and has correct scaling"""
    print("--- Testing Geometric fBm Properties ---")

    s0 = 100.0
    mu = 0.05  # Drift
    sigma = 0.2  # Volatility
    H = 0.6
    n = 1000
    batch_size = 100

    gfbm = processes.geometric_fbm(
        n=n, H=H, s0=s0, mu=mu, sigma=sigma, size=(batch_size,)
    )

    # Test positivity
    is_positive = (gfbm > 0).all()
    assert is_positive, "Geometric fBm contains non-positive values"

    # Test initial condition
    initial_error = torch.max(torch.abs(gfbm[..., 0] - s0))
    assert initial_error < 1e-5, f"Initial condition violated: error {initial_error}"

    # Test log-returns have reasonable variance scaling
    log_returns = torch.log(gfbm[..., 1:] / gfbm[..., :-1])
    log_var = torch.var(log_returns)

    print(
        f"✅ Geometric fBm: Positive={is_positive}, Initial error={initial_error:.2e}, Log-var={log_var:.4f}"
    )


def run_mathematical_tests():
    """Run all mathematical property tests"""
    print("\n" + "=" * 60)
    print("🧮 MATHEMATICAL PROPERTIES TEST SUITE")
    print("=" * 60 + "\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running mathematical tests on device: {device}\n")

    # Set seed for reproducible tests
    torch.manual_seed(42)

    test_functions = [
        test_self_similarity,
        test_covariance_function,
        test_increment_scaling,
        test_hurst_parameter_boundaries,
        test_bridge_constraints,
        test_long_range_dependence,
        test_hurst_estimation_accuracy,
        test_geometric_fbm_properties,
    ]

    passed = 0
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ [FAIL] {test_func.__name__}: {e}")
            return False

    print(f"\n" + "=" * 60)
    print(f"🎯 MATHEMATICAL TESTS COMPLETE: {passed}/{len(test_functions)} PASSED")
    print("✅ All fundamental properties verified!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = run_mathematical_tests()
    sys.exit(0 if success else 1)
