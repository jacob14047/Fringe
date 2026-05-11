"""
nuqkd.attacks.intercept_resend
================================
Intercept-and-Resend (IR) attack implementation.

Attack description
------------------
Eve intercepts a fraction ε of the photons travelling from Alice to Bob.
For each intercepted photon:

1. Eve measures it in a randomly chosen basis (Z or X).
2. Eve re-prepares a new photon in the *same state she measured* and sends it
   on to Bob.

Impact on QBER
~~~~~~~~~~~~~~
When Eve measures in the wrong basis (probability 1/2), the re-prepared
state is a superposition in Alice's original basis.  Bob then measures a
random result with probability 1/2.  The net QBER contribution is::

    QBER_IR = ε / 4

For ε = 1 (Eve intercepts everything): QBER ≈ 25 % >> 11 % threshold.
For ε = 0.44: QBER ≈ 11 % (the classical security bound for BB84).

Eve's information gain
~~~~~~~~~~~~~~~~~~~~~~
After the bases are announced on the classical channel, Eve knows which of
her measurements were in the correct basis.  For those (≈ ε/2 of the total
qubits), she has a perfect copy of the key bit.

Variants implemented
--------------------
* ``InterceptResendAttack`` — standard IR with configurable intercept rate.
* ``OptimalIRAttack`` — Eve measures in a breidbart basis to maximise
  information gain while minimising detectable QBER (symmetric attack).

References
----------
* Bennett & Brassard, 1984.
* Gisin et al., "Quantum Cryptography", Rev. Mod. Phys. 74, 145 (2002).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from nuqkd.attacks.base import BaseAttack
from nuqkd.core.classical import ClassicalMsgType
from nuqkd.core.qubit import (
    BASIS_X, BASIS_Z,
    Photon,
    measure_qubit,
    prepare_bb84_state,
)


class InterceptResendAttack(BaseAttack):
    """
    Standard Intercept-and-Resend attack.

    Parameters
    ----------
    intercept_rate : float
        ε ∈ [0, 1] — fraction of photons Eve intercepts.
    random_basis : bool
        If True Eve chooses her measurement basis uniformly at random
        (optimal in the information-theoretic sense).  If False she always
        measures in the Z basis (weaker, easier to detect).
    seed : int | None
        RNG seed for Eve (independent of Alice/Bob's RNG).
    """

    name = "intercept_resend"

    def __init__(self,
                 intercept_rate: float = 1.0,
                 random_basis: bool = True,
                 seed: Optional[int] = None) -> None:
        super().__init__()
        if not 0.0 <= intercept_rate <= 1.0:
            raise ValueError(f"intercept_rate must be in [0,1], got {intercept_rate}")
        self.intercept_rate = intercept_rate
        self.random_basis   = random_basis
        self._rng           = np.random.default_rng(seed)

        # Counters (accumulated across all iterations)
        self._total_intercepted: int      = 0
        self._correct_guesses:   int      = 0
        self._wrong_guesses:     int      = 0
        # Per-photon records: pulse_id → (eve_basis, eve_bit)
        self._interception_log: Dict[int, tuple] = {}

    # ------------------------------------------------------------------
    # Quantum hook
    # ------------------------------------------------------------------

    def intercept_quantum(self,
                          photons: List[Optional[Photon]]
                          ) -> List[Optional[Photon]]:
        """
        Intercept a fraction ``intercept_rate`` of the photons.

        For each intercepted photon:
        1. Measure in Eve's chosen basis.
        2. Re-prepare and return a new photon.
        """
        result: List[Optional[Photon]] = []
        for photon in photons:
            if photon is None:
                result.append(None)
                continue

            # Decide whether to intercept
            if photon.is_vacuum or float(self._rng.random()) >= self.intercept_rate:
                result.append(photon)
                continue

            # ---- Eve intercepts ------------------------------------------
            self._total_intercepted += 1

            # Choose Eve's measurement basis
            if self.random_basis:
                eve_basis = int(self._rng.integers(0, 2))
            else:
                eve_basis = BASIS_Z

            # Measure — Born rule
            eve_bit = measure_qubit(photon.state, eve_basis, self._rng)

            # Record interception
            self._interception_log[photon.pulse_id] = (eve_basis, eve_bit)

            # Re-prepare: send a photon in the *measured* state
            new_state = prepare_bb84_state(eve_basis, eve_bit)
            new_photon = photon.copy()
            new_photon.state           = new_state
            new_photon.was_intercepted = True
            new_photon.intercepted_by  = self.name

            result.append(new_photon)

        return result

    # ------------------------------------------------------------------
    # Classical hook
    # ------------------------------------------------------------------

    def listen_classical(self,
                         msg_type: ClassicalMsgType,
                         data: Any,
                         sender: str,
                         recipient: str) -> None:
        """
        After Alice announces bases, figure out which bits Eve got right.
        """
        super().listen_classical(msg_type, data, sender, recipient)

        if msg_type == ClassicalMsgType.ALICE_BASES and self.alice_bases_announced is not None:
            alice_bases = np.asarray(data)
            # Evaluate correctness of Eve's guesses
            for pulse_id, (eve_basis, eve_bit) in self._interception_log.items():
                if pulse_id < len(alice_bases):
                    if eve_basis == alice_bases[pulse_id]:
                        self._correct_guesses += 1
                    else:
                        self._wrong_guesses += 1

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def statistics(self) -> Dict[str, Any]:
        total = self._total_intercepted
        correct = self._correct_guesses
        info_fraction = correct / total if total > 0 else 0.0
        theoretical_qber = self.intercept_rate / 4.0
        return {
            "name":                 self.name,
            "intercept_rate":       self.intercept_rate,
            "total_intercepted":    total,
            "correct_guesses":      correct,
            "wrong_guesses":        self._wrong_guesses,
            "information_fraction": info_fraction,   # Eve's knowledge of sifted key
            "theoretical_qber_contribution": theoretical_qber,
        }

    def reset(self) -> None:
        super().reset()
        self._total_intercepted = 0
        self._correct_guesses   = 0
        self._wrong_guesses     = 0
        self._interception_log.clear()

    def on_iteration_start(self, iteration_id, config):
        super().on_iteration_start(iteration_id, config)
        self._interception_log.clear()


# ---------------------------------------------------------------------------
# Breidbart-basis (optimal symmetric) IR attack
# ---------------------------------------------------------------------------

class OptimalIRAttack(BaseAttack):
    """
    Optimal symmetric intercept-and-resend attack (Breidbart basis).

    Eve measures in the *Breidbart basis*::

        |ψ_0⟩ = cos(π/8)|H⟩ + sin(π/8)|V⟩
        |ψ_1⟩ = sin(π/8)|H⟩ − cos(π/8)|V⟩

    This maximises Eve's mutual information with Alice while keeping the QBER
    introduced as low as possible for a *symmetric* attack.

    Theoretical result:
    * QBER introduced per intercepted photon: (1 − 1/√2) / 2 ≈ 14.6 %
    * Information per intercepted photon: sin²(π/8) ≈ 0.146 bits

    This attack is undetectable if ε < ε* = 2(1 − 1/√2) ≈ 0.586 in the
    ideal, noiseless channel.

    Reference: Bruss, Phys. Rev. Lett. 81, 3018 (1998).
    """

    name = "optimal_ir_breidbart"

    def __init__(self,
                 intercept_rate: float = 1.0,
                 seed: Optional[int] = None) -> None:
        super().__init__()
        self.intercept_rate = intercept_rate
        self._rng = np.random.default_rng(seed)
        self._total_intercepted = 0

        # Breidbart projectors
        theta = np.pi / 8.0
        self._B0 = np.array([np.cos(theta), np.sin(theta)], dtype=complex)
        self._B1 = np.array([np.sin(theta), -np.cos(theta)], dtype=complex)

    def intercept_quantum(self, photons):
        result = []
        for photon in photons:
            if photon is None:
                result.append(None)
                continue
            if photon.is_vacuum or float(self._rng.random()) >= self.intercept_rate:
                result.append(photon)
                continue

            self._total_intercepted += 1

            # Born-rule measurement in Breidbart basis
            prob_b0 = float(abs(np.dot(self._B0.conj(), photon.state)) ** 2)
            if float(self._rng.random()) < prob_b0:
                new_state = self._B0.copy()
            else:
                new_state = self._B1.copy()

            new_photon            = photon.copy()
            new_photon.state      = new_state
            new_photon.was_intercepted = True
            new_photon.intercepted_by  = self.name
            result.append(new_photon)

        return result

    def statistics(self):
        return {
            "name":              self.name,
            "intercept_rate":    self.intercept_rate,
            "total_intercepted": self._total_intercepted,
            "theoretical_qber_per_photon": (1.0 - 1.0 / np.sqrt(2)) / 2.0,
        }

    def reset(self):
        super().reset()
        self._total_intercepted = 0
