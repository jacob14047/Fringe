"""
nuqkd.config.parameters
========================
Centralized, fully-typed configuration for the NuQKD simulation framework.

Every physical parameter the user might want to tune lives here as a frozen
dataclass.  The ``SimulationConfig`` root object is the single entry-point
that the rest of the library depends on – never global state.

Design intent
-------------
* **Modularity**: each sub-config maps 1-to-1 with a physical module so that
  future protocols (E91, SARG04, CV-QKD …) can reuse the unchanged modules.
* **Realism**: default values reflect a representative deployed system
  (SMF-28 fibre, InGaAs SPAD, 1550 nm).
* **Extensibility**: use ``extra_params: dict`` on any config to attach
  protocol-specific parameters without breaking the public API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ChannelType(str, Enum):
    OPTICAL_FIBER = "optical_fiber"
    FREE_SPACE    = "free_space"


class SourceType(str, Enum):
    IDEAL               = "ideal"       # Perfect single-photon source
    WEAK_COHERENT_PULSE = "wcp"         # Attenuated laser (Poissonian)


class RNGBackend(str, Enum):
    """Entropy source for bit/basis generation."""
    NUMPY  = "numpy"    # Seeded numpy Generator (reproducible)
    OS     = "os"       # os.urandom (cryptographically secure, not seedable)
    # ANU_QRNG = "anu_qrng"  # real quantum randomness (requires network)


class ErrorCorrectionScheme(str, Enum):
    NONE     = "none"
    CASCADE  = "cascade"
    LDPC     = "ldpc"
    IDEAL    = "ideal"    # Perfect correction; used for upper-bound analysis


# ---------------------------------------------------------------------------
# Sub-configurations
# ---------------------------------------------------------------------------

@dataclass
class SourceConfig:
    """
    Photon source model.

    For WCP sources the photon-number distribution of each pulse follows a
    Poisson distribution with mean ``mean_photon_number`` (μ).  Typical lab
    values: μ ∈ [0.1, 0.5].

    Decoy-state protocol
    --------------------
    Alice alternates between *signal* pulses (μ_s), *decoy* pulses (μ_d < μ_s),
    and optionally *vacuum* pulses (μ_v = 0).  The three intensities must
    satisfy μ_s > μ_d > μ_v ≥ 0.

    ``decoy_intensities``  : [μ_decoy, μ_vacuum]  (length matches decoy_probs)
    ``decoy_probabilities``: [p_decoy, p_vacuum]
    p_signal = 1 - sum(decoy_probabilities)
    """

    type: SourceType = SourceType.WEAK_COHERENT_PULSE

    # Signal pulse
    mean_photon_number: float = 0.5         # μ_s

    # Source clock (affects timing simulation)
    pulse_frequency_hz: float = 1.0e9       # 1 GHz — current state-of-the-art

    # Decoy-state options
    enable_decoy: bool = False
    # Default: one decoy + one vacuum (weak+vacuum protocol)
    decoy_intensities: List[float]   = field(default_factory=lambda: [0.1, 0.0])
    decoy_probabilities: List[float] = field(default_factory=lambda: [0.1, 0.05])

    extra_params: Dict = field(default_factory=dict)


@dataclass
class FiberChannelConfig:
    """Parameters specific to optical-fibre channels."""

    distance_km: float = 10.0

    # Loss
    attenuation_db_per_km: float = 0.2      # SMF-28 at 1550 nm: 0.18–0.20 dB/km

    # Dispersion (affects timing jitter at high rep-rates; not used by default)
    polarization_mode_dispersion_ps_per_sqrt_km: float = 0.1   # PMD coefficient
    chromatic_dispersion_ps_per_nm_km: float = 17.0            # CD at 1550 nm


@dataclass
class FreeSpaceChannelConfig:
    """Parameters specific to free-space optical (FSO) / satellite channels."""

    distance_km: float = 1.0
    wavelength_nm: float = 780.0            # Typical for free-space QKD

    # Atmospheric / geometric losses
    visibility_km: float = 10.0             # Meteorological visibility
    zenith_angle_deg: float = 0.0           # Elevation angle (0 = overhead)
    beam_divergence_mrad: float = 0.1       # Half-angle divergence
    aperture_diameter_m: float = 0.3        # Receiver telescope aperture

    # Tracking
    tracking_efficiency: float = 0.9        # Beam-pointing efficiency


@dataclass
class ChannelConfig:
    """
    Quantum channel — physical medium + noise model.

    The *depolarization* parameter ``p`` models the total probability that a
    photon undergoes one of the three Pauli errors (X, Y, Z), each with
    probability p/3 (isotropic depolarization channel).  This subsumes fibre
    birefringence fluctuations, polarisation-mode coupling, etc.

    ``insertion_loss_db`` captures all fixed losses not captured by the
    distance-dependent attenuation: splices, connectors, circulator, etc.
    """

    type: ChannelType = ChannelType.OPTICAL_FIBER

    fiber:      FiberChannelConfig      = field(default_factory=FiberChannelConfig)
    free_space: FreeSpaceChannelConfig  = field(default_factory=FreeSpaceChannelConfig)

    # Noise
    depolarization_prob: float = 0.01       # p ∈ [0, 1]

    # Fixed losses (connector, circulator, wavelength-division multiplexer…)
    insertion_loss_db: float = 3.0

    extra_params: Dict = field(default_factory=dict)


@dataclass
class DetectorConfig:
    """
    Single-photon detector model (SPAD / SNSPD).

    ``efficiency``          : η_D — probability that an arriving photon
                              produces a detection click.  SNSPD: up to 0.98;
                              InGaAs SPAD at 1550 nm: 0.10–0.30.
    ``dark_count_rate_hz``  : Background clicks in the absence of signal.
                              SNSPD: ~1 cps; SPAD: ~100–10000 cps.
    ``dead_time_ns``        : After each click the detector is blind for this
                              duration.  SPAD: ~20 000 ns; SNSPD: ~10–50 ns.
    ``timing_jitter_ps``    : Width (1σ) of the arrival-time uncertainty.
                              SNSPD: ~20 ps; SPAD: ~200–500 ps.
    ``afterpulse_prob``     : Probability that a true detection spawns a
                              spurious click in the next gate (SPAD artefact).
    """

    efficiency: float          = 0.85
    dark_count_rate_hz: float  = 100.0
    dead_time_ns: float        = 0.0       # 0 = module disabled
    timing_jitter_ps: float    = 50.0
    afterpulse_prob: float     = 0.005

    # Additional insertion loss at the detector face
    insertion_loss_db: float   = 3.0

    extra_params: Dict = field(default_factory=dict)


@dataclass
class ProtocolConfig:
    """
    Parameters governing the QKD protocol logic.

    These are *protocol-level* parameters, independent of the physical layer.
    A ``BB84`` implementation, an ``E91`` implementation, etc., all accept
    this same config object.
    """

    # --- Quantum phase ---
    raw_key_size: int   = 10_000    # N — photons Alice sends per round
    num_iterations: int = 10        # Number of independent key distributions

    # --- Classical post-processing ---
    qber_threshold: float   = 0.11  # Abort if estimated QBER > this value
    sharing_rate: float     = 0.50  # f — fraction of sifted key exposed for
                                    # QBER estimation (rest becomes secret key)

    # Error correction
    enable_error_correction: bool  = True
    ec_scheme: ErrorCorrectionScheme = ErrorCorrectionScheme.CASCADE
    # Efficiency penalty f_EC > 1 (ideal = 1, Cascade ≈ 1.16)
    ec_efficiency: float           = 1.16

    # Privacy amplification
    enable_privacy_amplification: bool = True
    security_epsilon: float        = 1e-10  # Composable security parameter ε

    # Finite-key corrections (Scarani-Renner, 2008)
    enable_finite_key_analysis: bool = False

    extra_params: Dict = field(default_factory=dict)


@dataclass
class AttackConfig:
    """
    Built-in eavesdropping configuration.

    These are the *default* attacks shipped with the library.  External attack
    agents (pentesters / AI) inject themselves through the ``BaseAttack``
    interface rather than through this config.
    """

    enabled: bool = False

    # Intercept-and-resend
    intercept_rate: float    = 0.0   # ε ∈ [0,1] — fraction of photons intercepted
    random_attacks: bool     = False  # Randomise which iterations are attacked
    attack_rate: float       = 0.5   # Fraction of iterations attacked (random mode)

    # Photon-Number Splitting
    enable_pns: bool = False

    extra_params: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Root configuration
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    """
    Root configuration object for a NuQKD simulation run.

    Usage example::

        from nuqkd.config import SimulationConfig, SourceConfig, ChannelConfig

        cfg = SimulationConfig(
            protocol_name="BB84",
            source=SourceConfig(mean_photon_number=0.5),
            channel=ChannelConfig(depolarization_prob=0.02),
        )

    All sub-configs are instantiated with their defaults if not provided.
    """

    protocol_name: str = "BB84"

    source:   SourceConfig   = field(default_factory=SourceConfig)
    channel:  ChannelConfig  = field(default_factory=ChannelConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    attack:   AttackConfig   = field(default_factory=AttackConfig)

    # Entropy source
    rng_backend: RNGBackend  = RNGBackend.NUMPY
    seed: Optional[int]      = None      # None → non-reproducible run

    # Research / debug flags
    enable_remaining_key_module: bool = False  # Compare QBER on unseen bits
    verbose: bool            = False
    export_results: bool     = True
    output_dir: str          = "./results"
