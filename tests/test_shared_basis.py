"""Shared POD must preserve a fixed mean/basis on held-out fields."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared_basis import SharedBasis


def test_known_subspace_heldout_and_roundtrip(tmp_path):
    mean = np.array([10., 20., 30.])
    direction = np.array([1., 2., 0.])
    train = mean + np.array([-2., -1., 1., 2.])[:, None]*direction
    basis = SharedBasis.fit(train, max_rank=3)
    assert basis.modes.shape == (3, 1)
    heldout = (mean+3*direction)[None]
    before = basis.mean.copy()
    np.testing.assert_allclose(basis.decode(basis.encode(heldout)), heldout, atol=1e-12)
    np.testing.assert_array_equal(basis.mean, before)
    np.testing.assert_allclose(basis.decode(basis.encode(heldout, rank=0)), mean[None])
    path = tmp_path/'basis.npz';basis.save(path)
    restored = SharedBasis.load(path)
    np.testing.assert_allclose(restored.decode(restored.encode(heldout)), heldout, atol=1e-12)


def test_unseen_direction_is_not_fitted():
    basis = SharedBasis.fit(np.array([[-1.,0.],[1.,0.]]), max_rank=1)
    heldout = np.array([[0.,5.]])
    np.testing.assert_allclose(basis.decode(basis.encode(heldout)), [[0.,0.]])


def test_reject_incompatible_and_nonfinite():
    basis = SharedBasis.fit(np.array([[-1.,0.],[1.,0.]]), max_rank=1)
    with pytest.raises(ValueError, match='grid shape'):
        basis.encode(np.ones((2,3)))
    with pytest.raises(ValueError, match='finite'):
        SharedBasis.fit([[1,np.nan],[2,3]], max_rank=1)
    with pytest.raises(ValueError, match='rank'):
        basis.encode([[1,2]], rank=2)
