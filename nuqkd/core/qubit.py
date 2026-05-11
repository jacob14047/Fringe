"""
nuqkd.core.qubit
=================
Quantum state representation for polarisation qubits.

Physical model
--------------
We use the **Jones-vector** formalism: a qubit is a two-component complex
vector |ψ⟩ = α|H⟩ + β|V⟩ with |α|² + |β|² = 1.

BB84 basis states
~~~~~~~~~~~~~~~~~
::

    Rectilinear (Z) basis:
        |0⟩_Z = |H⟩ = [1, 0]ᵀ      (horizontal, 0°)
        |1⟩_Z = |V⟩ = [0, 1]ᵀ      (vertical, 90°)

    Diagonal (X) basis:
        |0⟩_X = |+⟩ = [1, 1]ᵀ/√2   (+45°)
        |1⟩_X = |−⟩ = [1, −1]ᵀ/√2  (−45°)

Measurement (Born rule)
~~~~~~~~~~~~~~~~~~~~~~~
For a measurement in basis B with projectors {Π₀, Π₁}::

    P(outcome = k) = ⟨ψ|Πₖ|ψ⟩ = |⟨mₖ|ψ⟩|²

When Alice and Bob use different bases (Z vs X), the probability of any
particular outcome is exactly ½ — encoded information is randomised.

Depolarisation channel
~~~~~~~~~~~~~~~~~~~~~~
The isotropic depolarisation channel maps::

    ρ → (1 − p) ρ + (p/3)(X ρ X† + Y ρ Y† + Z ρ Z†)

In the pure-state simulation we apply one of the three Pauli errors with
probability p/3 each, leaving the state unchanged with probability 1 − p.

Note: p = 0.11 → QBER ≈ 11 % (the widely-cited unconditional security threshold
for BB84 under individual attacks is QBER < 11 %).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Basis and state constants
# ---------------------------------------------------------------------------

#: Rectilinear (Z / computational) basis index
BASIS_Z: int = 0
#: Diagonal (X / Hadamard) basis index
BASIS_X: int = 1

# Basis labels for display
BASIS_LABELS: Dict[int, str] = {BASIS_Z: "Z (rectilinear)", BASIS_X: "X (diagonal)"}

# The four BB84 polarisation states as Jones vectors
_INV_SQRT2 = 1.0 / np.sqrt(2.0)

STATE_H: np.ndarray = np.array([1.0, 0.0], dtype=complex)           # |H⟩
STATE_V: np.ndarray = np.array([0.0, 1.0], dtype=complex)           # |V⟩
STATE_P: np.ndarray = np.array([_INV_SQRT2, _INV_SQRT2], dtype=complex)   # |+⟩
STATE_M: np.ndarray = np.array([_INV_SQRT2, -_INV_SQRT2], dtype=complex)  # |−⟩

#: Lookup: (basis, bit) → Jones vector
BB84_STATES: Dict[Tuple[int, int], np.ndarray] = {
    (BASIS_Z, 0): STATE_H,
    (BASIS_Z, 1): STATE_V,
    (BASIS_X, 0): STATE_P,
    (BASIS_X, 1): STATE_M,
}

# Pauli matrices (used by the depolarisation channel)
PAULI_I: np.ndarray = np.eye(2, dtype=complex)
PAULI_X: np.ndarray = np.array([[0, 1], [1, 0]], dtype=complex)       # Bit-flip
PAULI_Y: np.ndarray = np.array([[0, -1j], [1j, 0]], dtype=complex)    # Bit+phase flip
PAULI_Z: np.ndarray = np.array([[1, 0], [0, -1]], dtype=complex)      # Phase-flip


# ---------------------------------------------------------------------------
# Core quantum operations
# ---------------------------------------------------------------------------

def prepare_bb84_state(basis: int, bit: int) -> np.ndarray:
    """
    Return the normalised Jones vector for a BB84 qubit.

    Parameters
    ----------
    basis : int
        0 → Z (rectilinear), 1 → X (diagonal).
    bit : int
        0 or 1.

    Returns
    -------
    np.ndarray shape (2,) complex
    """
    return BB84_STATES[(basis, bit)].copy()


def measure_qubit(state: np.ndarray, basis: int,
                  rng: np.random.Generator) -> int:
    """
    Perform a projective measurement on ``state`` in the given basis.

    The outcome probability follows the Born rule::

        P(k) = |⟨mₖ|ψ⟩|²

    Parameters
    ----------
    state : ndarray (2,) complex
        Jones vector of the incoming photon.
    basis : int
        Measurement basis (BASIS_Z or BASIS_X).
    rng : np.random.Generator
        Random number source for the stochastic collapse.

    Returns
    -------
    int
        Measurement outcome (0 or 1).
    """
    # Projection onto the |0⟩ eigenstate of the chosen basis
    ref_state_0 = STATE_H if basis == BASIS_Z else STATE_P
    amplitude   = np.dot(ref_state_0.conj(), state)
    prob_zero   = float(np.abs(amplitude) ** 2)
    return int(rng.random() >= prob_zero)   # 0 w.p. prob_zero, 1 otherwise


def apply_depolarization(state: np.ndarray, p: float,
                         rng: np.random.Generator) -> np.ndarray:
    """
    Apply the isotropic depolarisation channel to a pure qubit state.

    Each of the three Pauli errors (X, Y, Z) occurs with probability p/3.
    The state is unchanged with probability 1 − p.

    Parameters
    ----------
    state : ndarray (2,) complex
        Incoming qubit state.
    p : float
        Total depolarisation probability ∈ [0, 1].
    rng : np.random.Generator

    Returns
    -------
    ndarray (2,) complex
        Possibly-corrupted qubit state (still normalised).
    """
    r = float(rng.random())
    if r < p / 3.0:
        new_state = PAULI_X @ state   # Bit-flip (X error)
    elif r < 2.0 * p / 3.0:
        new_state = PAULI_Y @ state   # Bit+phase flip (Y error)
    elif r < p:
        new_state = PAULI_Z @ state   # Phase-flip (Z error)
    else:
        return state                  # No error
    # Re-normalise (numerical safety)
    norm = np.linalg.norm(new_state)
    return new_state / norm if norm > 0 else new_state


def state_fidelity(state_a: np.ndarray, state_b: np.ndarray) -> float:
    """Return |⟨ψ_a|ψ_b⟩|² ∈ [0, 1]."""
    return float(np.abs(np.dot(state_a.conj(), state_b)) ** 2)


# ---------------------------------------------------------------------------
# Data container for a photon in transit
# ---------------------------------------------------------------------------

@dataclass
class Photon:
    """
    A single photon (or WCS pulse) travelling through the quantum channel.

    This object carries *both* the quantum state and all the classical
    metadata needed for realistic simulation (timing, photon count, labels).

    Pentest / attack agents receive and return ``Photon`` objects from the
    quantum-channel hook — they may modify the state or drop the photon
    (return ``None`` to simulate blocking).

    Attributes
    ----------
    state : ndarray (2,) complex
        Jones vector.  Normalised on creation.
    basis : int
        Preparation basis chosen by Alice.
    bit_value : int
        Classical bit encoded by Alice.
    pulse_id : int
        Sequential index of the originating pulse.
    photon_count : int
        Number of photons in the pulse (WCS).  For ideal sources always 1.
    is_vacuum : bool
        True for zero-photon pulses (μ → 0 when they survive sifting stats).
    is_decoy : bool
        True for decoy-state pulses.
    intensity_label : str
        Human-readable label: "signal" | "decoy" | "vacuum".
    emission_time_ns : float
        Wall-clock time (ns) at which Alice emitted this pulse.
    arrival_time_ns : float
        Simulated arrival time at Bob's detector.
    was_intercepted : bool
        Set to True if any attack agent touched this photon.
    intercepted_by : str | None
        Name of the attack agent that last modified this photon.
    channel_errors : list[str]
        Log of physical errors applied (e.g. "X", "Y", "Z", "loss").
    """

    state:           np.ndarray
    basis:           int
    bit_value:       int
    pulse_id:        int
    photon_count:    int            = 1
    is_vacuum:       bool           = False
    is_decoy:        bool           = False
    intensity_label: str            = "signal"
    emission_time_ns: float         = 0.0
    arrival_time_ns:  float         = 0.0
    detection_time_ns: float        = 0.0

    # Attack telemetry (invisible to legitimate parties; for simulation analysis only)
    was_intercepted:  bool          = False
    intercepted_by:   Optional[str] = None
    channel_errors:   List[str]     = field(default_factory=list)

    def __post_init__(self) -> None:
        # Always work with a normalised copy
        self.state = np.asarray(self.state, dtype=complex)
        norm = np.linalg.norm(self.state)
        if norm > 1e-12:
            self.state = self.state / norm

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def create(cls,
               basis: int,
               bit: int,
               pulse_id: int,
               photon_count: int = 1,
               **kwargs) -> "Photon":
        """
        Construct a photon with the correct BB84 polarisation state.

        Parameters
        ----------
        basis, bit  : as described above.
        pulse_id    : originating pulse index.
        photon_count: number of photons in the pulse.
        **kwargs    : forwarded to ``__init__`` (e.g. ``is_decoy=True``).
        """
        state = prepare_bb84_state(basis, bit)
        return cls(
            state        = state,
            basis        = basis,
            bit_value    = bit,
            pulse_id     = pulse_id,
            photon_count = photon_count,
            **kwargs,
        )

    @classmethod
    def vacuum(cls, pulse_id: int, **kwargs) -> "Photon":
        """Create a vacuum (zero-photon) pulse placeholder."""
        return cls(
            state           = STATE_H.copy(),   # state irrelevant for vacuum
            basis           = BASIS_Z,
            bit_value       = 0,
            pulse_id        = pulse_id,
            photon_count    = 0,
            is_vacuum       = True,
            intensity_label = "vacuum",
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def copy(self) -> "Photon":
        """Deep copy (including the state array)."""
        import copy as _copy
        p = _copy.copy(self)
        p.state = self.state.copy()
        p.channel_errors = list(self.channel_errors)
        return p

    def apply_pauli(self, error: str) -> None:
        """
        In-place application of a Pauli error.

        Parameters
        ----------
        error : str
            One of ``"X"``, ``"Y"``, ``"Z"``.
        """
        ops = {"X": PAULI_X, "Y": PAULI_Y, "Z": PAULI_Z}
        if error not in ops:
            raise ValueError(f"Unknown Pauli error: {error!r}")
        self.state = ops[error] @ self.state
        norm = np.linalg.norm(self.state)
        if norm > 1e-12:
            self.state /= norm
        self.channel_errors.append(error)

    def measure(self, basis: int, rng: np.random.Generator) -> int:
        """
        Measure this photon in the given basis (Born rule).

        This is a *destructive* operation in reality, but here we keep
        the state unchanged so the simulation can track it.

        Returns
        -------
        int  —  0 or 1.
        """
        return measure_qubit(self.state, basis, rng)

    def __repr__(self) -> str:
        basis_str = "Z" if self.basis == BASIS_Z else "X"
        return (
            f"Photon(id={self.pulse_id}, basis={basis_str}, "
            f"bit={self.bit_value}, n={self.photon_count}, "
            f"decoy={self.is_decoy}, intercepted={self.was_intercepted})"
        )
