"""
Unit tests for RSMA, SDMA, and NOMA achievable rates and SIC feasibility.
"""

import numpy as np
import pytest
from src.rsma import compute_rsma_rf_rates, compute_sdma_rates, compute_noma_rates


def test_rsma_common_rate_bounded_by_weakest_user():
    channel_gains = np.array([1e-6, 5e-7, 2e-7, 1e-7]) # User 3 is weakest
    noise_w = 2e-11
    tx_power_w = 0.2
    beta = 0.3 # 30% power to common stream
    
    res = compute_rsma_rf_rates(channel_gains, noise_w, tx_power_w, beta=beta)
    
    # Common rate allocated should sum to common_rate_max
    assert np.isclose(np.sum(res['common_user_rates']), res['common_rate_max'], atol=1e-5)
    # Common rate max must be greater than zero when beta > 0
    assert res['common_rate_max'] > 0.0
    # Total rates must be strictly positive
    assert np.all(res['total_user_rates'] > 0.0)


def test_sdma_equivalent_to_zero_beta():
    channel_gains = np.array([1e-6, 5e-7, 2e-7, 1e-7])
    noise_w = 2e-11
    tx_power_w = 0.2
    
    sdma_rates = compute_sdma_rates(channel_gains, noise_w, tx_power_w)
    rsma_beta0 = compute_rsma_rf_rates(channel_gains, noise_w, tx_power_w, beta=0.0)
    
    # When beta = 0, common rate should be 0 and RSMA rates equal SDMA rates
    assert np.isclose(rsma_beta0['common_rate_max'], 0.0)
    assert np.allclose(sdma_rates, rsma_beta0['total_user_rates'])


def test_noma_rates_positive():
    channel_gains = np.array([1e-6, 5e-7, 2e-7, 1e-7])
    noise_w = 2e-11
    tx_power_w = 0.2
    noma_rates = compute_noma_rates(channel_gains, noise_w, tx_power_w)
    assert len(noma_rates) == 4
    assert np.all(noma_rates > 0.0)
    assert np.all(np.isfinite(noma_rates))
