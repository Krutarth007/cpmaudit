import numpy as np
import pytest

from cpmaudit.resampling import BootstrapSampler, out_of_bag_mask, subsample_indices


def test_cluster_bootstrap_keeps_clusters_whole():
    groups = np.repeat(np.arange(50), 3)
    sampler = BootstrapSampler("cluster", groups=groups)
    idx = sampler.draw(groups.size, np.random.default_rng(0))
    counts = np.bincount(groups[idx], minlength=50)
    assert np.all(counts % 3 == 0)
    assert sampler.n_clusters == 50


def test_stratified_bootstrap_keeps_event_count():
    y = np.r_[np.ones(20), np.zeros(80)].astype(int)
    idx = BootstrapSampler("stratified").draw(100, np.random.default_rng(1), y)
    assert y[idx].sum() == 20


def test_redraws_until_both_classes():
    y = np.zeros(200, dtype=int)
    y[0] = 1
    idx = BootstrapSampler("simple", max_tries=500).draw(200, np.random.default_rng(2), y)
    assert set(np.unique(y[idx])) == {0, 1}


def test_cluster_requires_groups():
    with pytest.raises(ValueError):
        BootstrapSampler("cluster")
    with pytest.raises(ValueError):
        BootstrapSampler("bogus")


def test_oob_and_subsample():
    assert out_of_bag_mask(5, np.array([0, 0, 2])).tolist() == [False, True, False, True, True]
    rng = np.random.default_rng(3)
    y = (rng.random(1000) < 0.2).astype(int)
    sub = subsample_indices(1000, 300, rng, y=y)
    assert abs(sub.size - 300) <= 2 and len(np.unique(sub)) == sub.size
    groups = np.repeat(np.arange(500), 2)
    sub = subsample_indices(1000, 300, rng, groups=groups)
    assert np.all(np.bincount(groups[sub])[np.unique(groups[sub])] == 2)
