"""
Unit tests for Performance Metrics (Jain's fairness, outage, coverage, latency).
"""

import numpy as np
import pytest
from src.metrics import (
    calculate_jains_fairness_index,
    calculate_outage_probability,
    calculate_coverage_probability,
    calculate_spectral_efficiency,
    calculate_latency_percentiles
)


def test_jains_fairness_equal_rates():
    # If all users have equal rates, fairness should be 1.0
    rates = np.array([50e6, 50e6, 50e6, 50e6])
    j = calculate_jains_fairness_index(rates)
    assert np.isclose(j, 1.0, atol=1e-5)


def test_jains_fairness_worst_case():
    # If 1 user gets all throughput and others get 0, fairness is 1/K = 0.25 for K=4
    rates = np.array([100e6, 0.0, 0.0, 0.0])
    j = calculate_jains_fairness_index(rates)
    assert np.isclose(j, 0.25, atol=1e-5)


def test_outage_and_coverage():
    rates = np.array([5e6, 15e6, 20e6, 2e6]) # 2 users below 10 Mbps
    min_rate = 10e6
    p_out = calculate_outage_probability(rates, min_rate)
    p_cov = calculate_coverage_probability(rates, min_rate)
    assert np.isclose(p_out, 0.5)
    assert np.isclose(p_cov, 0.5)
    assert np.isclose(p_out + p_cov, 1.0)


def test_latency_percentiles():
    lats = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    p = calculate_latency_percentiles(lats)
    assert p['mean'] == pytest.approx(0.55)
    assert p['p50'] == pytest.approx(0.55)
    assert p['p95'] >= 0.9
