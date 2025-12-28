#!/usr/bin/env python3
"""
Quick mathematical validation tests for torchfbm library.

Tests key properties:
1. Generator output shapes and bounds
2. H parameter boundary behavior
3. Bridge endpoint constraints
4. Hurst estimation accuracy
"""

import torch
import numpy as np
import sys
import os
from pathlib import Path

# Ensure we can import the local package
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

try:
    from torchfbm import generators, processes, estimators

    print("✅ Successfully imported torchfbm modules.")
except ImportError as e:
    print(f"❌ ImportError: {e}")
    sys.exit(1)


def test_mathematical_properties():
    """Test mathematical properties"""
    print("\n" + "=" * 50)
    print("🧮 MATHEMATICAL VALIDATION TESTS")
    print("=" * 50 + "\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}\n")
    torch.manual_seed(42)

    tests_passed = 0
    total_tests = 0

    # Test 1: H parameter boundaries
    print("--- Test 1: H Parameter Boundaries ---")
    total_tests += 1
    try:
        for H in [0.01, 0.1, 0.5, 0.9, 0.99]:
            fgn = generators.generate_davies_harte(n=500, H=H, size=(10,))
            assert torch.isfinite(fgn).all(), f"Non-finite values at H={H}"
        print("✅ H boundary test passed")
        tests_passed += 1
    except Exception as e:
        print(f"❌ H boundary test failed: {e}")

    # Test 2: Bridge constraints
    print("\n--- Test 2: Bridge Constraints ---")
    total_tests += 1
    try:
        start_val, end_val = -2.5, 4.1
        bridge = processes.fractional_brownian_bridge(
            n=200, H=0.6, start_val=start_val, end_val=end_val, size=(20,)
        )
        start_error = torch.max(torch.abs(bridge[..., 0] - start_val)).item()
        end_error = torch.max(torch.abs(bridge[..., -1] - end_val)).item()
        assert start_error < 1e-5, f"Bridge start error: {start_error}"
        assert end_error < 1e-5, f"Bridge end error: {end_error}"
        print(
            f"✅ Bridge constraints: Start error={start_error:.2e}, End error={end_error:.2e}"
        )
        tests_passed += 1
    except Exception as e:
        print(f"❌ Bridge constraints test failed: {e}")

    # Test 3: Hurst estimation accuracy
    print("\n--- Test 3: Hurst Estimation ---")
    total_tests += 1
    try:
        true_H_values = [0.3, 0.7]
        for true_H in true_H_values:
            fbm_path = generators.fbm(n=2000, H=true_H, size=(5,))
            estimated_H = estimators.estimate_hurst(fbm_path)
            mean_estimate = torch.mean(estimated_H).item()
            relative_error = abs(mean_estimate - true_H) / true_H
            print(
                f"    H={true_H}: Estimated={mean_estimate:.3f}, Error={relative_error:.3f}"
            )
            assert (
                relative_error < 0.2
            ), f"Hurst estimation error too large: {relative_error:.3f}"
        tests_passed += 1
        print("✅ Hurst estimation accuracy passed")
    except Exception as e:
        print(f"❌ Hurst estimation test failed: {e}")

    # Test 4: Geometric fBm positivity
    print("\n--- Test 4: Geometric fBm Properties ---")
    total_tests += 1
    try:
        s0 = 100.0
        gfbm = processes.geometric_fbm(n=500, H=0.5, s0=s0, size=(50,))
        is_positive = (gfbm > 0).all()
        initial_error = torch.max(torch.abs(gfbm[..., 0] - s0)).item()
        assert is_positive, "Geometric fBm contains non-positive values"
        assert initial_error < 1e-5, f"Initial condition error: {initial_error}"
        print(f"✅ Geometric fBm: All positive, Initial error={initial_error:.2e}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Geometric fBm test failed: {e}")

    # Test 5: Davies-Harte vs Cholesky consistency
    print("\n--- Test 5: Generator Consistency ---")
    total_tests += 1
    try:
        torch.manual_seed(123)  # Fixed seed for comparison
        H = 0.6
        n = 200

        # Generate multiple paths with same seed
        torch.manual_seed(123)
        fgn_dh = generators.generate_davies_harte(n=n, H=H, size=(10,))
        torch.manual_seed(123)
        fgn_chol = generators.generate_cholesky(n=n, H=H, size=(10,))

        # Compare statistical properties (not exact equality due to different algorithms)
        var_dh = torch.var(fgn_dh)
        var_chol = torch.var(fgn_chol)
        var_ratio = (var_dh / var_chol).item()

        # Should have similar variances (within 50% due to different methods)
        assert 0.5 < var_ratio < 2.0, f"Variance ratio out of range: {var_ratio:.3f}"
        print(f"✅ Generator consistency: Variance ratio DH/Chol = {var_ratio:.3f}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Generator consistency test failed: {e}")

    # Test 6: Self-similarity scaling
    print("\n--- Test 6: Self-Similarity Check ---")
    total_tests += 1
    try:
        H = 0.7
        n = 1000
        scale = 2.0

        # Original fBm
        fbm1 = generators.fbm(n=n, H=H, size=(100,))

        # Scaled fBm (double time points)
        fbm2 = generators.fbm(n=int(n * scale), H=H, size=(100,))
        fbm2_sampled = fbm2[..., :: int(scale)] / (scale**H)

        # Compare endpoint variances (should be similar)
        var1 = torch.var(fbm1[..., -1])
        var2 = torch.var(fbm2_sampled[..., -1])
        var_ratio = (var1 / var2).item()

        assert (
            0.7 < var_ratio < 1.3
        ), f"Self-similarity violated: variance ratio {var_ratio:.3f}"
        print(f"✅ Self-similarity: Variance ratio = {var_ratio:.3f}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Self-similarity test failed: {e}")

    print(f"\n" + "=" * 50)
    print(f"📊 MATHEMATICAL TESTS COMPLETE: {tests_passed}/{total_tests} PASSED")
    if tests_passed == total_tests:
        print("🎯 All mathematical properties verified!")
    else:
        print(f"⚠️  {total_tests - tests_passed} tests failed")
    print("=" * 50)

    return tests_passed == total_tests


if __name__ == "__main__":
    success = test_mathematical_properties()
    sys.exit(0 if success else 1)
