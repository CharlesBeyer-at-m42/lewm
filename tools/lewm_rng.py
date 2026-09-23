"""Bit-exact NumPy mirror of src/lewm/rng.cpp (splitmix64 counter stream, Box-Muller, inverse-CDF
Laplace, inverted dropout masks). Used by reference_run.py so both implementations consume the
same randomness. Vectorised over the counter; every draw is a pure function of (seed, stream, counter).
"""

from __future__ import annotations

import numpy as np

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)


def _splitmix64(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        x = x + _GOLDEN
        x = (x ^ (x >> np.uint64(30))) * _M1
        x = (x ^ (x >> np.uint64(27))) * _M2
        return x ^ (x >> np.uint64(31))


class Rng:
    def __init__(self, seed: int, stream: int):
        self.seed = np.uint64(seed)
        self.stream = np.uint64(stream)
        self.counter = 0

    def seek(self, counter: int) -> None:
        self.counter = int(counter)

    def _u64(self, n: int) -> np.ndarray:
        counters = np.arange(self.counter, self.counter + n, dtype=np.uint64)
        self.counter += n
        s = _splitmix64(np.array([self.seed], dtype=np.uint64))[0]
        s = _splitmix64(np.array([s ^ self.stream], dtype=np.uint64))[0]
        return _splitmix64(s ^ counters)

    def uniform(self, n: int) -> np.ndarray:
        bits = self._u64(n) >> np.uint64(11)
        return (bits.astype(np.float64) + 0.5) * (1.0 / 9007199254740992.0)

    def normal(self, n: int) -> np.ndarray:
        pairs = (n + 1) // 2
        u = self.uniform(2 * pairs)
        u1, u2 = u[0::2], u[1::2]
        r = np.sqrt(-2.0 * np.log(u1))
        theta = 2.0 * np.pi * u2
        out = np.empty(2 * pairs, dtype=np.float64)
        out[0::2] = r * np.cos(theta)
        out[1::2] = r * np.sin(theta)
        return out[:n].astype(np.float32)

    def laplace(self, n: int, scale: float) -> np.ndarray:
        u = self.uniform(n) - 0.5
        v = -scale * np.copysign(np.log(1.0 - 2.0 * np.abs(u)), u)
        return v.astype(np.float32)

    def dropout_mask(self, n: int, p: float) -> np.ndarray:
        keep = np.float32(1.0 / (1.0 - p))
        return np.where(self.uniform(n) >= p, keep, np.float32(0.0)).astype(np.float32)


# Stream ids shared with the C++ trainer (src/bin/lewm_train.cpp)
STREAM_PROJ = 1
STREAM_TARGET = 2
STREAM_DROPOUT = 3
