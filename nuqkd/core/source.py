"""
nuqkd.core.source
==================
Photon-source models for QKD simulations.

Two source types are provided:

IdealSource
    Emits exactly one photon per pulse.  Used as a theoretical baseline;
    immune to Photon-Number Splitting attacks.

WeakCoherentPulseSource (WCP)
    Attenuated laser producing pulses with a Poissonian photon-number
    distribution n ~ Poisson(μ_eff).  This is the de-facto standard source
    for practical QKD systems.

    The *effective* mean photon number μ_eff accounts for channel attenuation
    before the source generates the pulse list::

        μ_eff = μ_s · 10^(−(α·L + a_fixed) / 10)

    where α is the fibre attenuation coefficient (dB/km), L the distance, and
    a_fixed the fixed insertion losses.  The source itself always emits with μ_s;
    the channel attenuates photons probabilistically (see channel.py).

Decoy-state extension
---------------------
When decoy states are enabled, each pulse is independently labelled as
"signal", "decoy", or "vacuum" according to the configured probabilities.
Alice assigns a different mean photon number to each intensity level.
The intensity labels are recorded in each ``Photon`` object so the
post-processing step can perform decoy-state analysis.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from nuqkd.config.parameters import SourceConfig, SourceType
from nuqkd.core.qubit import BASIS_X, BASIS_Z, Photon
from nuqkd.utils.rng import NumpyRNG


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseSource(ABC):
    """Interface every photon source must implement."""

    def __init__(self, config: SourceConfig, rng: NumpyRNG) -> None:
        self.config = config
        self._rng   = rng

    @abstractmethod
    def emit_pulse(self,
                   pulse_id: int,
                   basis: int,
                   bit: int,
                   emission_time_ns: float = 0.0) -> Photon:
        """
        Emit one pulse encoding ``bit`` in ``basis``.

        Returns a ``Photon`` object whose ``photon_count`` reflects the
        source model (always 1 for ideal, Poisson-distributed for WCP).
        """

    def emit_sequence(self,
                      bases: np.ndarray,
                      bits: np.ndarray,
                      start_time_ns: float = 0.0) -> List[Photon]:
        """
        Emit N pulses according to ``bases`` and ``bits``.

        The inter-pulse spacing is ``1 / pulse_frequency_hz`` converted to ns.

        Returns a list of ``Photon`` objects (some may be vacuum pulses for WCP).
        """
        n = len(bases)
        period_ns = 1.0e9 / self.config.pulse_frequency_hz   # ns per clock slot
        photons: List[Photon] = []

        for i in range(n):
            t = start_time_ns + i * period_ns
            photon = self.emit_pulse(
                pulse_id=i,
                basis=int(bases[i]),
                bit=int(bits[i]),
                emission_time_ns=t,
            )
            photons.append(photon)

        return photons

    # ------------------------------------------------------------------
    # Decoy-state helpers  (called by both source subclasses)
    # ------------------------------------------------------------------

    def _sample_intensity(self) -> Tuple[float, str, bool]:
        """
        Sample the intensity for one pulse according to the decoy schedule.

        Returns
        -------
        (mean_photon_number, intensity_label, is_decoy) : tuple
        """
        cfg = self.config
        if not cfg.enable_decoy:
            return cfg.mean_photon_number, "signal", False

        p_decoys = np.asarray(cfg.decoy_probabilities)
        p_signal = 1.0 - float(p_decoys.sum())
        thresholds = np.concatenate([[0.0],
                                     np.cumsum(np.append(p_signal, p_decoys))])
        r = float(self._rng.random(1)[0])

        if r < thresholds[1]:
            return cfg.mean_photon_number, "signal", False
        for k, mu_d in enumerate(cfg.decoy_intensities):
            if thresholds[k + 1] <= r < thresholds[k + 2]:
                label    = "vacuum" if mu_d == 0.0 else "decoy"
                is_decoy = True
                return mu_d, label, is_decoy

        # Fallback (should not occur due to floating-point rounding at most)
        return cfg.mean_photon_number, "signal", False


# ---------------------------------------------------------------------------
# Ideal single-photon source
# ---------------------------------------------------------------------------

class IdealSource(BaseSource):
    """
    Deterministic single-photon emitter.

    Every pulse contains exactly one photon.  This source is immune to
    Photon-Number Splitting attacks but is currently unrealisable with
    room-temperature devices.
    """

    def emit_pulse(self,
                   pulse_id: int,
                   basis: int,
                   bit: int,
                   emission_time_ns: float = 0.0) -> Photon:
        mu, label, is_decoy = self._sample_intensity()
        # For an ideal source we always emit exactly 1 photon, but if decoy
        # mode labels this slot as a vacuum, we emit 0.
        photon_count = 0 if label == "vacuum" else 1
        return Photon.create(
            basis           = basis,
            bit             = bit,
            pulse_id        = pulse_id,
            photon_count    = photon_count,
            is_vacuum       = photon_count == 0,
            is_decoy        = is_decoy,
            intensity_label = label,
            emission_time_ns= emission_time_ns,
        )


# ---------------------------------------------------------------------------
# Weak Coherent Pulse (WCP) source
# ---------------------------------------------------------------------------

class WeakCoherentPulseSource(BaseSource):
    """
    Attenuated laser source — the standard practical source for QKD.

    The number of photons in each pulse is Poisson-distributed::

        n ~ Poisson(μ)

    where μ is the mean photon number.  Typical values: μ ∈ [0.1, 0.5].

    Photon-number statistics
    ~~~~~~~~~~~~~~~~~~~~~~~~
    * P(n = 0) = e^{−μ}          (vacuum pulse — carries no information)
    * P(n = 1) = μ e^{−μ}        (single-photon pulse — desired regime)
    * P(n ≥ 2) = 1 − (1+μ) e^{−μ} (multiphoton — PNS attack risk)

    For μ = 0.1:
        P(n=0) ≈ 90.5 %,  P(n=1) ≈ 9.0 %,  P(n≥2) ≈ 0.5 %

    Timing model
    ~~~~~~~~~~~~
    If the source needs to emit ``N`` photons and the pulse frequency is
    ``f_rep``, the expected generation time is (N / (μ · f_rep)) seconds,
    because each pulse contributes on average μ photons.  Vacuum pulses cost
    one clock cycle without contributing any qubit.
    """

    def emit_pulse(self,
                   pulse_id: int,
                   basis: int,
                   bit: int,
                   emission_time_ns: float = 0.0) -> Photon:
        mu, label, is_decoy = self._sample_intensity()

        # Sample photon count from Poisson(μ)
        n = int(self._rng.poisson(mu, size=1)[0])

        is_vacuum = (n == 0) or (label == "vacuum")

        return Photon.create(
            basis            = basis,
            bit              = bit,
            pulse_id         = pulse_id,
            photon_count     = n,
            is_vacuum        = is_vacuum,
            is_decoy         = is_decoy,
            intensity_label  = label,
            emission_time_ns = emission_time_ns,
        )

    # ------------------------------------------------------------------
    # Timing analysis helpers
    # ------------------------------------------------------------------

    def expected_time_for_n_qubits_ns(self, n_qubits: int,
                                      mu: Optional[float] = None) -> float:
        """
        Expected wall-clock time to generate ``n_qubits`` non-vacuum pulses.

        Each clock tick produces a non-vacuum pulse with probability
        1 − e^{−μ}.  The expected number of ticks needed is therefore
        N / (1 − e^{−μ}).

        Returns time in nanoseconds.
        """
        mu = mu or self.config.mean_photon_number
        p_non_vacuum = 1.0 - np.exp(-mu)
        ticks_needed = n_qubits / p_non_vacuum
        return ticks_needed * (1.0e9 / self.config.pulse_frequency_hz)

    def multiphoton_fraction(self, mu: Optional[float] = None) -> float:
        """
        Return the theoretical fraction of pulses containing ≥ 2 photons.

        These pulses are vulnerable to PNS attacks.
        """
        mu = mu or self.config.mean_photon_number
        return 1.0 - (1.0 + mu) * np.exp(-mu)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_source(config: SourceConfig, rng: NumpyRNG) -> BaseSource:
    """Instantiate the correct source implementation from config."""
    if config.type == SourceType.IDEAL:
        return IdealSource(config, rng)
    elif config.type == SourceType.WEAK_COHERENT_PULSE:
        return WeakCoherentPulseSource(config, rng)
    else:
        raise ValueError(f"Unknown source type: {config.type!r}")
