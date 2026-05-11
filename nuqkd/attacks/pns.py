"""
nuqkd.attacks.pns
==================
Photon-Number Splitting (PNS) attack implementation.

Attack description
------------------
The PNS attack exploits the multi-photon pulses produced by a Weak Coherent
Pulse (WCS) source.

For a pulse containing **n ≥ 2 photons**:

1. Eve **splits off one photon** and stores it in a lossless quantum memory.
   The remaining n−1 photons are forwarded to Bob.
2. Eve connects Alice and Bob via a **lossless channel** (replaces the lossy
   channel entirely) to avoid raising suspicion from anomalous photon
   statistics.
3. After the basis announcement on the classical channel, Eve measures the
   stored photon in Alice's basis — obtaining a perfect copy of the key bit.

For **single-photon** pulses Eve must choose a strategy that doesn't raise
the loss above the expected channel loss.  She either:

* Blocks a fraction of single-photon pulses (increasing apparent loss).
* Forwards the rest without interception.

The blocking fraction is chosen so that Bob's overall detection rate is
consistent with the legitimate channel loss expected by Alice and Bob.

Decoy-state detection
~~~~~~~~~~~~~~~~~~~~~
If Alice uses decoy states, she sends pulses with two (or three) different
intensities.  Since Eve cannot distinguish intensities, she applies the same
PNS strategy to all pulses.  The detection rate of *decoy* pulses will then
deviate from the expected rate, revealing Eve's presence.

Reference
---------
* Brassard et al., "Limitations on Practical Quantum Cryptography",
  PRL 85, 1330 (2000).
* Hwang, "Quantum Key Distribution with High Loss: Toward Global Secure
  Communication", PRL 91, 057901 (2003).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from nuqkd.attacks.base import BaseAttack
from nuqkd.core.classical import ClassicalMsgType
from nuqkd.core.qubit import Photon, measure_qubit


class PNSAttack(BaseAttack):
    """
    Photon-Number Splitting attack.

    Parameters
    ----------
    channel_transmittance : float
        T ∈ (0,1] — the legitimate channel transmittance Eve replaces.
        Eve uses a lossless channel for forwarding, so she must block
        single-photon pulses at rate (1 − T) to preserve statistics.
    store_fraction : float
        Fraction of multi-photon pulses Eve actually exploits (0 = passive,
        1 = fully exploit all multi-photon events).
    seed : int | None
    """

    name = "pns"

    def __init__(self,
                 channel_transmittance: float = 0.1,
                 store_fraction: float = 1.0,
                 seed: Optional[int] = None) -> None:
        super().__init__()
        if not 0.0 < channel_transmittance <= 1.0:
            raise ValueError("channel_transmittance must be in (0, 1]")
        self.T              = channel_transmittance
        self.store_fraction = store_fraction
        self._rng           = np.random.default_rng(seed)

        # Quantum memory: pulse_id → stolen photon state
        self._quantum_memory: Dict[int, np.ndarray] = {}

        # Counters
        self._multi_photon_events: int  = 0
        self._stored_photons: int       = 0
        self._blocked_single: int       = 0
        self._total_forwarded: int      = 0
        self._correct_bits_after_sift: int = 0

        # From classical channel
        self._alice_bases: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Quantum hook
    # ------------------------------------------------------------------

    def intercept_quantum(self,
                          photons: List[Optional[Photon]]
                          ) -> List[Optional[Photon]]:
        """
        PNS strategy:

        * n ≥ 2: steal one photon (store in memory), forward n−1 photons.
          Set photon_count = n−1 (the forwarded portion).
        * n = 1: block with probability (1 − T) to mimic channel loss.
        * n = 0: forward unchanged (vacuum).
        """
        result: List[Optional[Photon]] = []

        for photon in photons:
            if photon is None:
                result.append(None)
                continue

            n = photon.photon_count

            if n == 0 or photon.is_vacuum:
                result.append(photon)
                continue

            if n >= 2:
                # Multi-photon pulse — PNS opportunity
                self._multi_photon_events += 1
                if float(self._rng.random()) < self.store_fraction:
                    # Steal one photon: store state in quantum memory
                    self._quantum_memory[photon.pulse_id] = photon.state.copy()
                    self._stored_photons += 1

                    # Forward with one fewer photon
                    fwd = photon.copy()
                    fwd.photon_count     = n - 1
                    fwd.was_intercepted  = True
                    fwd.intercepted_by   = self.name
                    fwd.is_vacuum        = (n - 1 == 0)
                    self._total_forwarded += 1
                    result.append(fwd if not fwd.is_vacuum else None)
                else:
                    result.append(photon)

            else:
                # Single-photon pulse: block w.p. (1 − T) to mimic channel loss
                if float(self._rng.random()) > self.T:
                    self._blocked_single += 1
                    result.append(None)
                else:
                    self._total_forwarded += 1
                    result.append(photon)

        return result

    # ------------------------------------------------------------------
    # Classical hook
    # ------------------------------------------------------------------

    def listen_classical(self,
                         msg_type: ClassicalMsgType,
                         data: Any,
                         sender: str,
                         recipient: str) -> None:
        super().listen_classical(msg_type, data, sender, recipient)

        if msg_type == ClassicalMsgType.ALICE_BASES:
            self._alice_bases = np.asarray(data)
            # Measure stored photons in the now-known basis
            self._measure_stored_photons()

        elif msg_type == ClassicalMsgType.SIFTED_INDICES:
            # Count how many of Eve's stored photons are in the sifted key
            sifted = np.asarray(data)
            self._correct_bits_after_sift = sum(
                1 for pid in self._quantum_memory if pid in set(sifted.tolist())
            )

    def _measure_stored_photons(self) -> None:
        """Measure all quantum-memory entries in Alice's announced basis."""
        if self._alice_bases is None:
            return
        for pulse_id, state in list(self._quantum_memory.items()):
            if pulse_id < len(self._alice_bases):
                # Eve now knows the correct basis
                basis = int(self._alice_bases[pulse_id])
                bit   = measure_qubit(state, basis, self._rng)
                # Replace state entry with resolved (basis, bit) pair
                self._quantum_memory[pulse_id] = np.array([basis, bit])

    # ------------------------------------------------------------------
    # Decoy detection oracle (for simulation analysis)
    # ------------------------------------------------------------------

    def detectable_by_decoy(self,
                            signal_vacancies: float,
                            decoy_vacancies: float,
                            threshold: float = 0.05) -> bool:
        """
        Heuristic: is this PNS attack detectable via decoy state analysis?

        Eve cannot distinguish signal from decoy pulses, so she applies the
        same blocking strategy.  The legitimate parties compare the vacancy
        rate of signal vs. decoy pulses; if they differ significantly, a PNS
        attack is flagged.

        Parameters
        ----------
        signal_vacancies : float
            Fraction of signal-pulse slots where Bob detected nothing.
        decoy_vacancies : float
            Same for decoy pulses.
        threshold : float
            Minimum difference that raises a flag.

        Returns
        -------
        bool  — True if the attack is likely detectable.
        """
        return abs(signal_vacancies - decoy_vacancies) > threshold

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def statistics(self) -> Dict[str, Any]:
        total = self._stored_photons + self._blocked_single
        return {
            "name":                     self.name,
            "channel_transmittance":    self.T,
            "multi_photon_events":      self._multi_photon_events,
            "stored_photons":           self._stored_photons,
            "blocked_single_photons":   self._blocked_single,
            "total_forwarded":          self._total_forwarded,
            "correct_bits_in_sifted":   self._correct_bits_after_sift,
        }

    def reset(self) -> None:
        super().reset()
        self._quantum_memory.clear()
        self._multi_photon_events  = 0
        self._stored_photons       = 0
        self._blocked_single       = 0
        self._total_forwarded      = 0
        self._correct_bits_after_sift = 0
        self._alice_bases          = None
