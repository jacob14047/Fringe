"""
nuqkd.core.channel
===================
Quantum channel models.

A quantum channel takes a list of photons emitted by Alice's source and
delivers a (possibly smaller, possibly modified) list to Bob's detector.

Physical effects modelled
--------------------------
1. **Distance-dependent attenuation**
   Each photon survives the channel with probability T (the transmittance)::

       T = 10^{−(α · L + a_fixed) / 10}

   where α [dB/km] is the attenuation coefficient, L [km] the link distance,
   and a_fixed [dB] the sum of fixed insertion losses.

2. **Isotropic depolarisation noise**
   Surviving photons may undergo a Pauli error with total probability p (see
   ``qubit.apply_depolarization``).

3. **Propagation delay**
   Photons arrive at Bob after the light-travel time::

       Δt = L / (c / n_eff)

   where n_eff ≈ 1.4678 for silica fibre at 1550 nm.

Attack injection
----------------
The channel maintains an optional *attack agent* slot.  If an agent is
registered (via ``register_attack``), the photon list is passed through
``agent.intercept_quantum(photons)`` *before* depolarisation is applied.
The agent may:

* Return the photon unmodified (transparent channel).
* Modify the qubit state (intercept-and-resend).
* Return ``None`` for a photon to block it (simulate PNS or channel loss
  introduced by Eve).

The physical noise is always applied *after* the attack — this models Eve
being located closer to Alice than to Bob, consistent with the standard BB84
security analysis.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from nuqkd.config.parameters import ChannelConfig, ChannelType
from nuqkd.core.qubit import Photon, apply_depolarization
from nuqkd.utils.rng import NumpyRNG

if TYPE_CHECKING:
    from nuqkd.attacks.base import BaseAttack

# Speed of light in fibre (m/s)  (c / n_eff, n_eff ≈ 1.4678 for SMF at 1550 nm)
C_FIBRE_M_PER_S: float = 2.0416e8
# Speed of light in vacuum (m/s)
C_VACUUM_M_PER_S: float = 2.9979e8


# ---------------------------------------------------------------------------
# Channel result container
# ---------------------------------------------------------------------------

class ChannelTransitResult:
    """
    Summary of one quantum-phase transmission.

    Attributes
    ----------
    photons_sent : int
        Number of pulses Alice emitted.
    photons_survived : int
        Photons that physically reached Bob's detector input face.
    photons_blocked : int
        Photons blocked by Eve (for PNS / IR attack accounting).
    transmission_delay_ns : float
        One-way propagation delay in nanoseconds.
    transmittance : float
        Effective channel transmittance T ∈ [0, 1].
    surviving_photons : list[Photon]
        Photons that made it through (post-attenuation, post-attack,
        pre-detector).
    """

    def __init__(self) -> None:
        self.photons_sent: int         = 0
        self.photons_survived: int     = 0
        self.photons_blocked_eve: int  = 0
        self.photons_lost_channel: int = 0
        self.transmission_delay_ns: float = 0.0
        self.transmittance: float      = 0.0
        self.surviving_photons: List[Photon] = []


# ---------------------------------------------------------------------------
# Abstract channel
# ---------------------------------------------------------------------------

class BaseQuantumChannel(ABC):
    """
    Abstract quantum channel.

    Subclasses implement ``_compute_transmittance`` and optionally
    ``_propagation_delay_ns``.

    The ``transmit`` method orchestrates the full pipeline:
    (1) attack interception, (2) photon-loss sampling, (3) depolarisation,
    (4) arrival-time stamping.
    """

    def __init__(self, config: ChannelConfig, rng: NumpyRNG) -> None:
        self.config  = config
        self._rng    = rng
        self._attack: Optional["BaseAttack"] = None

    # ------------------------------------------------------------------
    # Attack agent registration
    # ------------------------------------------------------------------

    def register_attack(self, attack: "BaseAttack") -> None:
        """
        Register an eavesdropping agent on this channel.

        Call with ``None`` to remove the active agent.
        """
        self._attack = attack

    @property
    def has_attack(self) -> bool:
        return self._attack is not None

    # ------------------------------------------------------------------
    # Physical parameters (must be implemented by subclasses)
    # ------------------------------------------------------------------

    @abstractmethod
    def _compute_transmittance(self) -> float:
        """Return the end-to-end photon survival probability T ∈ (0, 1]."""

    def _propagation_delay_ns(self) -> float:
        """Return one-way propagation delay in nanoseconds."""
        return 0.0

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def transmit(self, photons: List[Photon]) -> ChannelTransitResult:
        """
        Pass ``photons`` through the full channel pipeline.

        Pipeline order:

        1. Attack agent intercepts (if registered).
        2. Physical loss sampling (binomial — each photon survives w.p. T).
        3. Depolarisation noise on surviving photons.
        4. Timing stamps updated.

        Returns
        -------
        ChannelTransitResult
        """
        result = ChannelTransitResult()
        result.photons_sent     = len(photons)
        result.transmittance    = self._compute_transmittance()
        result.transmission_delay_ns = self._propagation_delay_ns()

        # ---- Step 1: Attack agent ----------------------------------------
        if self._attack is not None:
            photons = self._attack.intercept_quantum(photons)
            # Agent may return None entries for blocked photons
            blocked = sum(1 for p in photons if p is None)
            result.photons_blocked_eve = blocked
            photons = [p for p in photons if p is not None]

        # ---- Step 2: Physical channel attenuation ------------------------
        surviving: List[Photon] = []
        for photon in photons:
            if photon.is_vacuum:
                # Vacuum pulses don't carry photons — skip loss sampling
                # but keep for timing reference if needed
                continue
            survived = self._apply_loss(photon, result.transmittance)
            if survived:
                surviving.append(photon)
            else:
                photon.channel_errors.append("loss")
                result.photons_lost_channel += 1

        # ---- Step 3: Depolarisation noise --------------------------------
        p_dep = self.config.depolarization_prob
        if p_dep > 0.0:
            rng_np = self._rng.numpy_rng
            for photon in surviving:
                old_state   = photon.state.copy()
                new_state   = apply_depolarization(photon.state, p_dep, rng_np)
                photon.state = new_state
                if not np.allclose(old_state, new_state):
                    # Record which Pauli error occurred (inferred from state change)
                    photon.channel_errors.append("depol")

        # ---- Step 4: Timing ----------------------------------------------
        delay_ns = result.transmission_delay_ns
        for photon in surviving:
            photon.arrival_time_ns = photon.emission_time_ns + delay_ns

        result.photons_survived   = len(surviving)
        result.surviving_photons  = surviving
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_loss(self, photon: Photon, T: float) -> bool:
        """
        Decide whether a photon survives the channel.

        For a pulse with n photons, each survives independently w.p. T.
        The probability that at least one photon reaches Bob is::

            P(survive) = 1 − (1 − T)^n

        In the WCP regime with n=1 (the dominant case) this simplifies to T.
        """
        n = photon.photon_count
        if n == 0:
            return False
        p_survive = 1.0 - (1.0 - T) ** n
        return float(self._rng.random(1)[0]) < p_survive

    def effective_mu(self, mu_source: float) -> float:
        """Return μ_eff = μ_source · T (for WCS timing analysis)."""
        return mu_source * self._compute_transmittance()

    def secret_key_rate_upper_bound(self, mu: float,
                                    qber: float,
                                    eta_d: float) -> float:
        """
        Simplified upper bound on the secret key rate (bits per pulse)
        using the GLLP / Shor-Preskill formula for WCS with single-photon
        component::

            R ≤ Q_1 · [1 − h(e_1)] − Q_μ · f · h(e_μ)

        This is only an indicative bound; exact values require decoy analysis.
        """
        T = self._compute_transmittance()
        # Gain from single-photon component (approximate)
        Q1  = mu * np.exp(-mu) * T * eta_d
        Qmu = (1.0 - np.exp(-mu * T * eta_d))
        h   = lambda e: (-e * np.log2(e + 1e-12) - (1 - e) * np.log2(1 - e + 1e-12))
        e1  = qber                # approximate
        return max(0.0, Q1 * (1.0 - h(e1)) - Qmu * 1.16 * h(qber))

    def __repr__(self) -> str:
        T = self._compute_transmittance()
        return (
            f"{self.__class__.__name__}("
            f"T={T:.4f}, "
            f"dep={self.config.depolarization_prob:.3f}, "
            f"attack={self.has_attack})"
        )


# ---------------------------------------------------------------------------
# Optical-fibre channel
# ---------------------------------------------------------------------------

class OpticalFiberChannel(BaseQuantumChannel):
    """
    SMF-based quantum channel.

    Transmittance
    ~~~~~~~~~~~~~
    ::

        T = 10^{−(α · L + a_fixed + a_det) / 10}

    where:
        α       = attenuation coefficient [dB/km]
        L       = distance [km]
        a_fixed = fixed insertion losses (connectors, WDM, etc.) [dB]
        a_det   = detector coupling loss [dB]  (from DetectorConfig — passed
                  separately to avoid circular imports)

    Propagation delay
    ~~~~~~~~~~~~~~~~~
    ::

        Δt = L [km] × 1000 / C_FIBRE [m/s] × 1e9  (converted to ns)
    """

    def __init__(self, config: ChannelConfig, rng: NumpyRNG,
                 detector_insertion_loss_db: float = 3.0) -> None:
        super().__init__(config, rng)
        self._detector_loss_db = detector_insertion_loss_db

    def _compute_transmittance(self) -> float:
        fc  = self.config.fiber
        total_loss_db = (
            fc.attenuation_db_per_km * fc.distance_km
            + self.config.insertion_loss_db
            + self._detector_loss_db
        )
        return 10.0 ** (-total_loss_db / 10.0)

    def _propagation_delay_ns(self) -> float:
        distance_m = self.config.fiber.distance_km * 1e3
        return distance_m / C_FIBRE_M_PER_S * 1e9


# ---------------------------------------------------------------------------
# Free-space channel
# ---------------------------------------------------------------------------

class FreeSpaceChannel(BaseQuantumChannel):
    """
    Free-space optical (FSO) / satellite quantum channel.

    Loss model
    ~~~~~~~~~~
    Total loss is the product of:

    * **Geometric / diffraction loss** (beam divergence over distance)::

          L_geo = (θ · L / D_rx)²

      where θ = half-angle divergence, L = distance, D_rx = receiver aperture.

    * **Atmospheric extinction** (Beer-Lambert in a turbulent atmosphere).
      We use the simplified model::

          T_atm = exp(−L_km / V_km)

      where V_km is the meteorological visibility.

    * **Tracking efficiency** η_track.

    The total transmittance is::

        T = η_track · T_atm · (D_rx / (2 · θ · L))²

    clamped to (0, 1].
    """

    def __init__(self, config: ChannelConfig, rng: NumpyRNG,
                 detector_insertion_loss_db: float = 3.0) -> None:
        super().__init__(config, rng)
        self._det_loss_db = detector_insertion_loss_db

    def _compute_transmittance(self) -> float:
        fs    = self.config.free_space
        L_m   = fs.distance_km * 1e3

        # Geometric loss
        theta_rad = fs.beam_divergence_mrad * 1e-3
        D_rx_m    = fs.aperture_diameter_m
        denom     = 2.0 * theta_rad * L_m
        if denom <= 0:
            T_geo = 1.0
        else:
            T_geo = min(1.0, (D_rx_m / denom) ** 2)

        # Atmospheric extinction
        V_km  = max(fs.visibility_km, 1e-6)
        T_atm = np.exp(-fs.distance_km / V_km)

        # Detector coupling
        T_det = 10.0 ** (-self._det_loss_db / 10.0)

        T = fs.tracking_efficiency * T_atm * T_geo * T_det
        return float(np.clip(T, 1e-12, 1.0))

    def _propagation_delay_ns(self) -> float:
        distance_m = self.config.free_space.distance_km * 1e3
        return distance_m / C_VACUUM_M_PER_S * 1e9


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_channel(config: ChannelConfig, rng: NumpyRNG,
                   detector_insertion_loss_db: float = 3.0) -> BaseQuantumChannel:
    """Instantiate the correct channel from configuration."""
    if config.type == ChannelType.OPTICAL_FIBER:
        return OpticalFiberChannel(config, rng, detector_insertion_loss_db)
    elif config.type == ChannelType.FREE_SPACE:
        return FreeSpaceChannel(config, rng, detector_insertion_loss_db)
    else:
        raise ValueError(f"Unknown channel type: {config.type!r}")
