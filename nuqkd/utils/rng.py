"""
nuqkd.utils.rng
================
Unified random-number generation interface.

The simulation needs truly independent, unbiased bit/basis sequences.  We
support three backends:

* ``NUMPY``  — numpy's PCG64 DXSM generator; seedable for reproducibility.
* ``OS``     — draws from the OS entropy pool (``os.urandom``); suitable for
               realistic security benchmarks, but not reproducible.

The ANU QRNG live feed can be added here later without touching any other
module (just add a new ``RNGBackend`` variant and the corresponding class).
"""

from __future__ import annotations

import os
import struct
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from nuqkd.config.parameters import RNGBackend


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseRNG(ABC):
    """Common interface for all entropy sources."""

    @abstractmethod
    def integers(self, low: int, high: int, size: int) -> np.ndarray:
        """Return ``size`` integers uniformly drawn from [low, high)."""

    @abstractmethod
    def random(self, size: int = 1) -> np.ndarray:
        """Return ``size`` floats uniformly drawn from [0, 1)."""

    @abstractmethod
    def choice(self, a: np.ndarray, size: int, replace: bool = True) -> np.ndarray:
        """Uniformly sample from ``a``."""

    def bits(self, size: int) -> np.ndarray:
        """Convenience: ``size`` uniformly distributed bits (0 or 1)."""
        return self.integers(0, 2, size)

    def bases(self, size: int) -> np.ndarray:
        """Convenience: ``size`` uniformly distributed bases (0 or 1)."""
        return self.integers(0, 2, size)


# ---------------------------------------------------------------------------
# Numpy backend
# ---------------------------------------------------------------------------

class NumpyRNG(BaseRNG):
    """
    Wraps ``numpy.random.Generator`` (PCG64-DXSM by default).

    This is the recommended backend for reproducible simulation runs.
    """

    def __init__(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)

    @property
    def numpy_rng(self) -> np.random.Generator:
        """Expose underlying numpy RNG for modules that need it directly."""
        return self._rng

    def integers(self, low: int, high: int, size: int) -> np.ndarray:
        return self._rng.integers(low, high, size=size)

    def random(self, size: int = 1) -> np.ndarray:
        return self._rng.random(size)

    def choice(self, a: np.ndarray, size: int, replace: bool = True) -> np.ndarray:
        return self._rng.choice(a, size=size, replace=replace)

    def poisson(self, lam: float, size: int = 1) -> np.ndarray:
        """Poisson-distributed integers (photon counts per pulse)."""
        return self._rng.poisson(lam, size=size)

    def normal(self, loc: float, scale: float, size: int = 1) -> np.ndarray:
        """Gaussian-distributed values (timing jitter)."""
        return self._rng.normal(loc, scale, size=size)

    def uniform(self, low: float = 0.0, high: float = 1.0,
                size: int = 1) -> np.ndarray:
        return self._rng.uniform(low, high, size=size)


# ---------------------------------------------------------------------------
# OS entropy backend
# ---------------------------------------------------------------------------

class OsRNG(BaseRNG):
    """
    Draws from the operating-system entropy pool (``/dev/urandom`` on Linux,
    ``CryptGenRandom`` on Windows).

    NOT seedable — each run is unique.  Use for security-critical benchmarks.
    """

    def _raw_bytes(self, n: int) -> bytes:
        return os.urandom(n)

    def integers(self, low: int, high: int, size: int) -> np.ndarray:
        # We draw uniform bytes and map to [low, high)
        span = high - low
        raw = np.frombuffer(self._raw_bytes(size * 4), dtype=np.uint32)
        return (raw % span + low).astype(np.int64)

    def random(self, size: int = 1) -> np.ndarray:
        raw = np.frombuffer(self._raw_bytes(size * 4), dtype=np.uint32)
        return raw.astype(np.float64) / 2**32

    def choice(self, a: np.ndarray, size: int, replace: bool = True) -> np.ndarray:
        indices = self.integers(0, len(a), size)
        return a[indices]

    def poisson(self, lam: float, size: int = 1) -> np.ndarray:
        # Fall back to numpy for Poisson — we seed it fresh each call
        return np.random.default_rng(
            int.from_bytes(os.urandom(8), "little")
        ).poisson(lam, size=size)

    def normal(self, loc: float, scale: float, size: int = 1) -> np.ndarray:
        return np.random.default_rng(
            int.from_bytes(os.urandom(8), "little")
        ).normal(loc, scale, size=size)

    def uniform(self, low: float = 0.0, high: float = 1.0,
                size: int = 1) -> np.ndarray:
        return self.random(size) * (high - low) + low


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_rng(backend: RNGBackend, seed: Optional[int] = None) -> NumpyRNG | OsRNG:
    """Instantiate the requested RNG backend."""
    if backend == RNGBackend.NUMPY:
        return NumpyRNG(seed)
    elif backend == RNGBackend.OS:
        if seed is not None:
            import warnings
            warnings.warn(
                "OsRNG does not support seeding; seed value ignored.",
                UserWarning,
                stacklevel=2,
            )
        return OsRNG()
    else:
        raise ValueError(f"Unknown RNG backend: {backend!r}")
