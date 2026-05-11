"""
nuqkd.protocols.bb84
=====================
BB84 protocol implementation — one complete key-distribution round.

Orchestrates the pipeline:

  Alice                 Quantum Channel              Bob
  ─────                 ───────────────              ───
  generate bits/bases → emit photons → [attack?] → detect → measure bases

  ─── Classical Channel (authenticated, public) ───
  Bob  → announces detections
  Alice→ announces bases for detected slots
  Bob  → announces his bases
  Both → sift (keep matching bases)
  Both → QBER estimation + abort if needed
  Both → error correction
  Both → privacy amplification → secret key

The protocol is deliberately decoupled from the physical layer:
swap ``SourceConfig``, ``ChannelConfig``, ``DetectorConfig`` to
simulate different hardware without touching this class.

Vulnerability injection
-----------------------
The ``VulnerabilityProfile`` is passed to the ``ChannelSession`` before
the protocol runs.  It modifies:
  * Source photon statistics   (mu_excess, weak_rng)
  * Timing of emitted photons  (basis_dependent_timing, decoy_timing_leak)
  * Detector behaviour         (efficiency_mismatch, blinding_vulnerable)
  * Post-processing            (pa_seed_leakage)

None of these modifications are visible through the protocol's public API —
they surface only as statistical anomalies in the ``ChannelObserver``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from nuqkd.config.parameters import SimulationConfig
from nuqkd.config.vulnerability_profiles import VulnerabilityProfile
from nuqkd.core.channel import BaseQuantumChannel, create_channel
from nuqkd.core.classical import ClassicalChannel, ClassicalMsgType
from nuqkd.core.detector import DetectionRecord, SinglePhotonDetector
from nuqkd.core.qubit import BASIS_X, BASIS_Z, Photon
from nuqkd.core.source import BaseSource, create_source
from nuqkd.postprocessing import (
    PostProcessingResult,
    run_postprocessing,
)
from nuqkd.utils.rng import NumpyRNG, create_rng

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container for one protocol iteration
# ---------------------------------------------------------------------------

@dataclass
class BB84IterationResult:
    """
    Complete record of one BB84 key-distribution round.

    All arrays have length ``n_pulses`` (the raw key size).
    """

    iteration_id:   int
    n_pulses:       int

    # ---- Alice's raw data ------------------------------------------------
    alice_bits:   np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint8))
    alice_bases:  np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint8))
    pulse_types:  np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    # Effective μ per pulse (may differ from declared if mu_excess active)
    pulse_mus:    np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    # Emission timestamps [ns]
    emit_times:   np.ndarray = field(default_factory=lambda: np.array([], dtype=float))

    # ---- Bob's raw data --------------------------------------------------
    bob_bits:     np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint8))
    bob_bases:    np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint8))
    detected:     np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    # Detection timestamps [ns] (includes jitter)
    detect_times: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    # Per-slot click type: 'genuine' | 'dark' | 'afterpulse' | 'none'
    click_types:  np.ndarray = field(default_factory=lambda: np.array([], dtype=object))

    # ---- Channel telemetry -----------------------------------------------
    photons_sent:      int   = 0
    photons_survived:  int   = 0
    photons_blocked_eve: int = 0
    channel_T:         float = 0.0
    attack_active:     bool  = False

    # ---- Post-processing -------------------------------------------------
    postproc:  Optional[PostProcessingResult] = None

    # ---- Timing ----------------------------------------------------------
    wall_time_s: float = 0.0

    # ---- Convenience properties ------------------------------------------

    @property
    def n_detected(self) -> int:
        return int(np.sum(self.detected))

    @property
    def n_sifted(self) -> int:
        if self.postproc is None:
            return 0
        return len(self.postproc.sifting.sifted_alice)

    @property
    def qber(self) -> float:
        if self.postproc is None:
            return float("nan")
        return self.postproc.qber.estimated_qber

    @property
    def secret_key_bits(self) -> int:
        if self.postproc is None:
            return 0
        return self.postproc.pa.secret_key_bits

    @property
    def aborted(self) -> bool:
        if self.postproc is None:
            return True
        return self.postproc.aborted

    @property
    def detection_rate(self) -> float:
        return self.n_detected / max(self.n_pulses, 1)

    def summary(self) -> Dict[str, Any]:
        return {
            "iteration":        self.iteration_id,
            "n_pulses":         self.n_pulses,
            "n_detected":       self.n_detected,
            "detection_rate":   round(self.detection_rate, 4),
            "n_sifted":         self.n_sifted,
            "qber":             round(self.qber, 4) if not np.isnan(self.qber) else None,
            "secret_key_bits":  self.secret_key_bits,
            "aborted":          self.aborted,
            "attack_active":    self.attack_active,
            "photons_survived": self.photons_survived,
            "channel_T":        round(self.channel_T, 6),
            "wall_time_s":      round(self.wall_time_s, 3),
        }


# ---------------------------------------------------------------------------
# BB84 Protocol
# ---------------------------------------------------------------------------

class BB84Protocol:
    """
    One complete BB84 key-distribution session.

    Parameters
    ----------
    config : SimulationConfig
        Full hardware + protocol configuration.
    vuln_profile : VulnerabilityProfile
        Hidden vulnerability profile injected into the channel.
        Agents must *discover* its effects through observation.
    attack_agent : BaseAttack | None
        If provided, wired into the quantum and classical channels.
    seed : int | None
        Master RNG seed (for reproducibility).
    """

    def __init__(self,
                 config: SimulationConfig,
                 vuln_profile: Optional[VulnerabilityProfile] = None,
                 attack_agent=None,
                 seed: Optional[int] = None) -> None:
        self.config      = config
        self.vuln        = vuln_profile or VulnerabilityProfile()
        self.attack      = attack_agent

        # Master RNG — Alice, Bob, and channel each get independent streams
        master_seed = seed if seed is not None else config.seed
        self._master_rng = create_rng(config.rng_backend, master_seed)

        # Child RNGs (derived from master for reproducibility)
        self._alice_rng  = NumpyRNG(self._derive_seed(master_seed, 0))
        self._bob_rng    = NumpyRNG(self._derive_seed(master_seed, 1))
        self._channel_rng = NumpyRNG(self._derive_seed(master_seed, 2))

        # Physical layer
        slot_ns = 1.0e9 / config.source.pulse_frequency_hz

        self._source   = create_source(config.source, self._alice_rng)
        self._channel  = create_channel(
            config.channel,
            self._channel_rng,
            detector_insertion_loss_db=config.detector.insertion_loss_db,
        )
        self._detector = SinglePhotonDetector(
            config.detector,
            self._bob_rng,
            slot_duration_ns=slot_ns,
        )
        self._classical = ClassicalChannel()

        # Wire attack agent
        if self.attack is not None:
            self._channel.register_attack(self.attack)
            self._classical.register_attack(self.attack)

        self._iteration = 0

    # ------------------------------------------------------------------
    # Single iteration
    # ------------------------------------------------------------------

    def run_iteration(self) -> BB84IterationResult:
        """
        Execute one complete BB84 key-distribution round.

        Returns a fully-populated ``BB84IterationResult``.
        """
        t0  = time.perf_counter()
        cfg = self.config.protocol
        N   = cfg.raw_key_size

        result = BB84IterationResult(
            iteration_id = self._iteration,
            n_pulses     = N,
            attack_active = self.attack is not None,
        )

        if self.attack is not None:
            self.attack.on_iteration_start(self._iteration, self.config)

        # ── Phase 1: Alice prepares ────────────────────────────────────────
        alice_bits, alice_bases = self._alice_prepare(N)
        result.alice_bits  = alice_bits
        result.alice_bases = alice_bases

        # ── Phase 2: Emit photons ──────────────────────────────────────────
        photons = self._source.emit_sequence(alice_bases, alice_bits)
        self._apply_timing_vuln(photons, alice_bases)     # hidden side-channel
        result.emit_times  = np.array([p.emission_time_ns for p in photons])
        result.pulse_types = np.array([p.intensity_label  for p in photons])
        result.pulse_mus   = np.array([p.photon_count      for p in photons],
                                       dtype=float)

        # ── Phase 3: Quantum channel transmission ──────────────────────────
        transit = self._channel.transmit(photons)
        result.photons_sent       = transit.photons_sent
        result.photons_survived   = transit.photons_survived
        result.photons_blocked_eve = transit.photons_blocked_eve
        result.channel_T          = transit.transmittance

        # Build a slot-indexed lookup: pulse_id → surviving photon
        survived_map = {p.pulse_id: p for p in transit.surviving_photons}

        # ── Phase 4: Bob measures ──────────────────────────────────────────
        bob_bases = self._bob_choose_bases(N)
        result.bob_bases = bob_bases

        bob_bits      = np.zeros(N, dtype=np.uint8)
        detected      = np.zeros(N, dtype=bool)
        detect_times  = np.zeros(N, dtype=float)
        click_types   = np.full(N, "none", dtype=object)

        self._detector.reset()
        slot_ns = 1.0e9 / self.config.source.pulse_frequency_hz

        for i in range(N):
            photon     = survived_map.get(i, None)
            t_arrival  = (photon.arrival_time_ns if photon is not None
                          else i * slot_ns + transit.transmission_delay_ns)

            rec = self._detector.detect(photon, int(bob_bases[i]), t_arrival)

            if rec.clicked:
                detected[i]      = True
                bob_bits[i]      = rec.bob_bit
                detect_times[i]  = rec.detection_time_ns
                click_types[i]   = ("genuine" if rec.is_genuine
                                    else "dark" if rec.is_dark
                                    else "afterpulse")

        result.bob_bits     = bob_bits
        result.detected     = detected
        result.detect_times = detect_times
        result.click_types  = click_types

        # ── Phase 5: Classical post-processing ─────────────────────────────
        result.postproc = self._run_classical(
            alice_bits, alice_bases, bob_bits, bob_bases, detected
        )

        result.wall_time_s = time.perf_counter() - t0

        if self.attack is not None:
            self.attack.on_iteration_end(self._iteration, result)

        self._iteration += 1
        return result

    # ------------------------------------------------------------------
    # Multi-iteration run
    # ------------------------------------------------------------------

    def run(self, n_iterations: Optional[int] = None) -> List[BB84IterationResult]:
        """
        Run ``n_iterations`` independent key-distribution rounds.

        Returns a list of ``BB84IterationResult`` objects, one per round.
        """
        n = n_iterations or self.config.protocol.num_iterations
        results = []
        for i in range(n):
            if self.config.verbose:
                logger.info("[BB84] iteration %d/%d", i + 1, n)
            r = self.run_iteration()
            results.append(r)
            if self.config.verbose:
                logger.info("  %s", r.summary())
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _alice_prepare(self, N: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Alice generates N random bits and N random bases.

        If ``weak_rng`` vulnerability is active, the basis sequence is
        biased and/or periodic — but generated here without announcement.
        """
        bits = self._alice_rng.bits(N)

        if self.vuln.weak_rng:
            bases = self._biased_periodic_bases(N)
        else:
            bases = self._alice_rng.bases(N)

        return bits.astype(np.uint8), bases.astype(np.uint8)

    def _biased_periodic_bases(self, N: int) -> np.ndarray:
        """Generate biased + possibly periodic basis sequence (hidden vuln)."""
        p_z    = self.vuln.rng_bias_z
        period = self.vuln.rng_period
        rng    = self._alice_rng.numpy_rng

        if period > 0 and period < N:
            # Generate one period then tile
            one_period = (rng.random(period) < p_z).astype(np.uint8)
            repeated   = np.tile(one_period, N // period + 1)[:N]
            return repeated
        else:
            return (rng.random(N) < p_z).astype(np.uint8)

    def _bob_choose_bases(self, N: int) -> np.ndarray:
        """Bob chooses measurement bases uniformly at random."""
        return self._bob_rng.bases(N).astype(np.uint8)

    def _apply_timing_vuln(self, photons: List[Photon],
                            bases: np.ndarray) -> None:
        """
        Inject basis-dependent timing offset into photon emission times.

        This is a hidden vulnerability — not announced via any public API.
        Z-basis photons arrive ``timing_delta_ps_z_vs_x`` ps earlier than
        X-basis photons.
        """
        if not self.vuln.basis_dependent_timing:
            return

        delta_ns = self.vuln.timing_delta_ps_z_vs_x / 1000.0
        rng      = self._channel_rng.numpy_rng

        for photon, basis in zip(photons, bases):
            if basis == BASIS_X:
                photon.emission_time_ns += delta_ns
            # Add per-basis extra jitter
            extra_jitter_ps = self.vuln.timing_jitter_per_basis.get(int(basis), 0.0)
            if extra_jitter_ps > 0.0:
                photon.emission_time_ns += float(
                    rng.normal(0.0, extra_jitter_ps / 1000.0)
                )

        # Decoy timing offset (separate vulnerability)
        if self.vuln.decoy_timing_leak:
            offset_ns = self.vuln.decoy_timing_offset_ps / 1000.0
            for photon in photons:
                if photon.is_decoy:
                    photon.emission_time_ns += offset_ns

    def _run_classical(self,
                        alice_bits:  np.ndarray,
                        alice_bases: np.ndarray,
                        bob_bits:    np.ndarray,
                        bob_bases:   np.ndarray,
                        detected:    np.ndarray) -> PostProcessingResult:
        """
        Execute the classical channel exchange and post-processing.

        Every message goes through the ClassicalChannel, which delivers it
        to Eve's ``listen_classical`` hook before the recipient sees it.
        """
        cc = self._classical

        # 1. Bob announces which slots had a detection
        detected_indices = np.where(detected)[0].tolist()
        cc.send("Bob", "Alice", ClassicalMsgType.BOB_DETECTIONS, detected_indices)

        # 2. Alice announces her bases for those slots
        alice_bases_detected = alice_bases.tolist()
        cc.send("Alice", "Bob", ClassicalMsgType.ALICE_BASES, alice_bases_detected)

        # 3. Bob announces his bases
        bob_bases_list = bob_bases.tolist()
        cc.send("Bob", "Alice", ClassicalMsgType.BOB_BASES, bob_bases_list)

        # 4. Run full post-processing pipeline
        pp_result = run_postprocessing(
            alice_bits   = alice_bits,
            alice_bases  = alice_bases,
            bob_bits     = bob_bits,
            bob_bases    = bob_bases,
            detected_mask = detected,
            config       = self.config.protocol,
            rng          = self._alice_rng.numpy_rng,
        )

        # 5. Announce sifted indices (Eve hears this)
        sifted_idx = pp_result.sifting.sifted_indices.tolist()
        cc.send("Alice", "Bob", ClassicalMsgType.SIFTED_INDICES, sifted_idx)

        # 6. Announce QBER estimate
        cc.send("Alice", "Bob", ClassicalMsgType.QBER_ESTIMATE,
                pp_result.qber.estimated_qber)

        if pp_result.aborted:
            cc.send("Alice", "Bob", ClassicalMsgType.PROTOCOL_ABORT,
                    pp_result.abort_reason)

        return pp_result

    @staticmethod
    def _derive_seed(master: Optional[int], offset: int) -> Optional[int]:
        if master is None:
            return None
        return (master * 6364136223846793005 + offset) & 0xFFFF_FFFF_FFFF_FFFF
