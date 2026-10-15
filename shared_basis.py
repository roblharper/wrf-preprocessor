"""Fixed POD basis fitted across training cases on a common local grid.

This experimental encoder does not alter conditional_cases.generate. Flattened
field features require consistent grid ordering across training and evaluation.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SharedBasis:
    mean: np.ndarray
    modes: np.ndarray
    field_shape: tuple

    @classmethod
    def fit(cls, fields, *, max_rank):
        """Fit one mean and orthonormal feature basis from training cases only."""
        data = np.asarray(fields, dtype=np.float64)
        if data.ndim < 2 or len(data) < 2 or max_rank < 0:
            raise ValueError('Require at least two training fields and a nonnegative rank.')
        if not np.isfinite(data).all():
            raise ValueError('Training fields must be finite.')
        shape = data.shape[1:]
        flat = data.reshape(len(data), -1)
        mean = flat.mean(axis=0)
        _, singular, vt = np.linalg.svd(flat-mean, full_matrices=False)
        tolerance = np.finfo(np.float64).eps * max(flat.shape) * (singular[0] if len(singular) else 0)
        rank = min(max_rank, int(np.count_nonzero(singular > tolerance)))
        return cls(mean, vt[:rank].T.copy(), shape)

    def encode(self, fields, *, rank=None):
        data = np.asarray(fields, dtype=np.float64)
        if data.shape[1:] != self.field_shape or not np.isfinite(data).all():
            raise ValueError('Fields must be finite and match the fitted grid shape.')
        rank = self.modes.shape[1] if rank is None else rank
        if rank < 0 or rank > self.modes.shape[1]:
            raise ValueError('Requested rank exceeds fitted rank.')
        return (data.reshape(len(data), -1)-self.mean) @ self.modes[:, :rank]

    def decode(self, coefficients):
        coefficients = np.asarray(coefficients, dtype=np.float64)
        if coefficients.ndim != 2 or coefficients.shape[1] > self.modes.shape[1] or not np.isfinite(coefficients).all():
            raise ValueError('Invalid shared-basis coefficients.')
        flat = self.mean + coefficients @ self.modes[:, :coefficients.shape[1]].T
        return flat.reshape((len(coefficients), *self.field_shape))

    def save(self, path):
        np.savez_compressed(Path(path), mean=self.mean, modes=self.modes,
                            field_shape=np.asarray(self.field_shape, dtype=np.int64))

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            mean, modes = data['mean'], data['modes']
            shape = tuple(int(n) for n in data['field_shape'])
        if (mean.shape != (int(np.prod(shape)),) or modes.ndim != 2 or modes.shape[0] != len(mean)
                or not np.isfinite(mean).all() or not np.isfinite(modes).all()):
            raise ValueError('Invalid shared-basis artifact.')
        return cls(mean, modes, shape)
