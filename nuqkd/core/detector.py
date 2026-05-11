"""
nuqkd.core.detector
====================
Single-photon detector (SPD) simulation.

Physical effects modelled
--------------------------
* **Quantum efficiency** η_D — probability that an arriving photon triggers
  a click.  For multi-photon pulses (WCS), the effective detection probability
  is::

      P_det(n) = 1 − (1 − η_D)^n

  (at least one photon of the n arriving triggers the detector).

* **Dark counts** — thermally / electrically generated spontaneous clicks at
  rate d_c [Hz].  In each time slot of duration Δt [s]::

      P_dark = d_c · Δt

  Dark clicks are assigned a random basis and bit value — they add noise
  without Alice's knowledge.

* **Dead time** τ [ns] — after a click the detector is unresponsive for τ ns.
  If a photon arrives during the dead time it is silently dropped.

* **Afterpulsing** — with probability p_ap, each genuine click spawns a
  spurious click in the *next* time slot.  Afterpulses carry no photon
  information; they are assigned a random basis/bit.

* **Timing jitter** — the recorded detection time is perturbed by
  Gaussian noise N(0, σ_jitter) where σ_jitter = timing_jitter_ps / 1000 [ns].

Detection record
~~~~~~~~~~~~~~~~
Each call to ``detect`` returns a ``DetectionRecord`` capturing:

* Whether a genuine click occurred.
* Whether a dark count occurred.
* The recorded detection time.
* The measured bit (if click).

These records drive the post-processing pipeline (sifting, QBER estimation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from nuqkd.config.parameters import DetectorConfig
from nuqkd.core.qubit import Photon, measure_qubit
from nuqkd.utils.rng import NumpyRNG


# ---------------------------------------------------------------------------
# Detection record
# ---------------------------------------------------------------------------

@dataclass
class DetectionRecord:
    """
    Result of one detector evaluation for a single pulse slot.

    Attributes
    ----------
    pulse_id : int
        The ``Photon.pulse_id`` from the originating pulse.
    clicked : bool
        True if *any* detection event occurred (genuine + dark + afterpulse).
    is_genuine : bool
        True if the click was caused by an actual photon.
    is_dark : bool
        True if the click was a dark count.
    is_afterpulse : bool
        True if the click is an afterpulse from the previous detection.
    bob_basis : int
        Measurement basis Bob chose for this slot (0 = Z, 1 = X).
    bob_bit : int
        Bob's measurement result (0 or 1).  Meaningless if ``clicked`` is False.
    detection_time_ns : float
        Recorded detection timestamp (including jitter).
    was_during_dead_time : bool
        Photon arrived but was discarded because detector was inactive.
    """

    pulse_id:            int
    clicked:             bool   = False
    is_genuine:          bool   = False
    is_dark:             bool   = False
    is_afterpulse:       bool   = False
    bob_basis:           int    = 0
    bob_bit:             int    = 0
    detection_time_ns:   float  = 0.0
    was_during_dead_time: bool  = False

    def __repr__(self) -> str:
        if not self.clicked:
            return f"DetectionRecord(id={self.pulse_id}, no click)"
        kind = "genuine" if self.is_genuine else ("dark" if self.is_dark else "afterpulse")
        return (
            f"DetectionRecord(id={self.pulse_id}, {kind}, "
            f"basis={'Z' if self.bob_basis == 0 else 'X'}, "
            f"bit={self.bob_bit})"
        )


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class SinglePhotonDetector:
    """
    Time-gated single-photon detector.

    The detector evaluates one pulse slot at a time (``detect`` method).
    It maintains internal state to model dead time and afterpulsing across
    consecutive calls.

    Parameters
    ----------
    config : DetectorConfig
    rng    : NumpyRNG
    slot_duration_ns : float
        Duration of each time slot (= 1 / f_rep [ns]).  Used to compute the
        dark-count probability per slot.
    """

    def __init__(self,
                 config: DetectorConfig,
                 rng: NumpyRNG,
                 slot_duration_ns: float = 1.0) -> None:
        self.config           = config
        self._rng             = rng
        self._slot_ns         = slot_duration_ns

        # State
        self._dead_until_ns:  float = 0.0   # time when dead time expires
        self._last_clicked:   bool  = False  # for afterpulse modelling
        self._current_time_ns: float = 0.0

    # ------------------------------------------------------------------
    # Core detection logic
    # ------------------------------------------------------------------

    def detect(self,
               photon: Optional[Photon],
               bob_basis: int,
               pulse_arrival_time_ns: float) -> DetectionRecord:
        """
        Evaluate one time slot.

        Parameters
        ----------
        photon : Photon | None
            Photon arriving in this slot.  ``None`` means no photon arrived
            (vacuum or fully attenuated).
        bob_basis : int
            Measurement basis Bob randomly chose for this slot.
        pulse_arrival_time_ns : float
            Nominal arrival time of the photon (before jitter).

        Returns
        -------
        DetectionRecord
        """
        pulse_id = photon.pulse_id if photon is not None else -1
        rec = DetectionRecord(pulse_id=pulse_id, bob_basis=bob_basis)
        rec.detection_time_ns = pulse_arrival_time_ns

        self._current_time_ns = pulse_arrival_time_ns
        rng = self._rng.numpy_rng

        # ---- Afterpulse check (from the *previous* slot) -----------------
        if self._last_clicked and self.config.afterpulse_prob > 0.0:
            if float(rng.random()) < self.config.afterpulse_prob:
                rec.clicked      = True
                rec.is_afterpulse = True
                rec.bob_bit       = int(rng.integers(0, 2))
                rec.detection_time_ns += self._jitter_ns(rng)
                self._last_clicked = True
                return rec  # afterpulse dominates this slot

        # Reset afterpulse flag; set again below if we click
        self._last_clicked = False

        # ---- Dead time check ---------------------------------------------
        if (self.config.dead_time_ns > 0.0 and
                pulse_arrival_time_ns < self._dead_until_ns):
            rec.was_during_dead_time = True
            # Can we still get a dark count?  No — detector is fully blind.
            return rec

        # ---- Genuine photon detection ------------------------------------
        if photon is not None and not photon.is_vacuum and photon.photon_count > 0:
            p_det = self._detection_probability(photon.photon_count)
            if float(rng.random()) < p_det:
                rec.clicked    = True
                rec.is_genuine = True
                # Measure the arriving qubit state
                rec.bob_bit    = measure_qubit(photon.state, bob_basis, rng)
                rec.detection_time_ns = (
                    pulse_arrival_time_ns + self._jitter_ns(rng)
                )
                photon.detection_time_ns = rec.detection_time_ns
                self._register_click(rec.detection_time_ns)
                return rec

        # ---- Dark count --------------------------------------------------
        p_dark = self.config.dark_count_rate_hz * (self._slot_ns * 1e-9)
        if float(rng.random()) < p_dark:
            rec.clicked = True
            rec.is_dark = True
            rec.bob_bit = int(rng.integers(0, 2))
            rec.detection_time_ns += self._jitter_ns(rng)
            self._register_click(rec.detection_time_ns)

        return rec

    # ------------------------------------------------------------------
    # Batch detection
    # ------------------------------------------------------------------

    def detect_sequence(self,
                        photons_by_slot: List[Optional[Photon]],
                        bob_bases: np.ndarray,
                        start_time_ns: float = 0.0) -> List[DetectionRecord]:
        """
        Detect a sequence of pulses.

        Parameters
        ----------
        photons_by_slot : list of Photon | None
            Index i → photon in slot i (None if nothing arrived).
        bob_bases : ndarray of int
            Bob's measurement bases (one per slot).
        start_time_ns : float
            Absolute time of slot 0.

        Returns
        -------
        list of DetectionRecord (one per slot).
        """
        records: List[DetectionRecord] = []
        for i, (photon, basis) in enumerate(zip(photons_by_slot, bob_bases)):
            t_arrival = start_time_ns + i * self._slot_ns
            if photon is not None:
                t_arrival = photon.arrival_time_ns
            records.append(self.detect(photon, int(basis), t_arrival))
        return records

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _detection_probability(self, n_photons: int) -> float:
        """
        P(detect | n photons arrive).

        Each photon triggers the detector independently with probability η_D.
        Detection happens if at least one photon triggers it::

            P(detect) = 1 − (1 − η_D)^n
        """
        return 1.0 - (1.0 - self.config.efficiency) ** n_photons

    def _jitter_ns(self, rng: np.random.Generator) -> float:
        """Sample timing jitter (Gaussian, σ = timing_jitter_ps / 1000)."""
        sigma_ns = self.config.timing_jitter_ps / 1000.0
        return float(rng.normal(0.0, sigma_ns)) if sigma_ns > 0 else 0.0

    def _register_click(self, t_ns: float) -> None:
        """Update dead-time timer and afterpulse flag."""
        self._dead_until_ns = t_ns + self.config.dead_time_ns
        self._last_clicked  = True

    def reset(self) -> None:
        """Reset internal state (call between independent key distributions)."""
        self._dead_until_ns  = 0.0
        self._last_clicked   = False
        self._current_time_ns = 0.0

    # ------------------------------------------------------------------
    # Statistics helpers
    # ------------------------------------------------------------------

    def expected_dark_count_prob_per_slot(self) -> float:
        return self.config.dark_count_rate_hz * self._slot_ns * 1e-9

    def __repr__(self) -> str:
        return (
            f"SinglePhotonDetector("
            f"η={self.config.efficiency:.2f}, "
            f"d_c={self.config.dark_count_rate_hz:.0f} Hz, "
            f"τ={self.config.dead_time_ns:.0f} ns, "
            f"jitter={self.config.timing_jitter_ps:.0f} ps)"
        )
