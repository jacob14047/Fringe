"""
nuqkd.core.session
===================
``QKDSession`` — the single entry-point for running a complete simulation.

This class wires together every module:

  SimulationConfig + VulnerabilityProfile
        │
        ▼
  BB84Protocol ─────────── ChannelObserver
        │                       │
        ▼                       ▼
  BB84IterationResult    MeasurementResult
        │
        ▼
  SessionMetrics ──────── coherence_check()
        │
        ▼
  SessionReport  (JSON-serialisable)

Usage (minimal)::

    from nuqkd.core.session import QKDSession
    from nuqkd.config.vulnerability_profiles import PROFILE_MEDIUM

    session = QKDSession.from_profile("medium", seed=42)
    report  = session.run()
    print(report.summary())

Usage (custom)::

    from nuqkd.config import SimulationConfig, DetectorConfig
    from nuqkd.config.vulnerability_profiles import VulnerabilityProfile

    cfg = SimulationConfig(verbose=True)
    cfg.detector = DetectorConfig(efficiency=0.95, dark_count_rate_hz=10)

    vuln = VulnerabilityProfile(
        basis_dependent_timing=True,
        timing_delta_ps_z_vs_x=45.0,
    )

    session = QKDSession(cfg, vuln, seed=123)
    report  = session.run()
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from nuqkd.config.parameters import SimulationConfig
from nuqkd.config.vulnerability_profiles import (
    NAMED_PROFILES,
    VulnerabilityProfile,
    get_random_profile,
)
from nuqkd.core.channel import create_channel
from nuqkd.core.channel_observer import ChannelObserver
from nuqkd.metrics.collector import SessionMetrics
from nuqkd.protocols.bb84 import BB84IterationResult, BB84Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session report
# ---------------------------------------------------------------------------

@dataclass
class SessionReport:
    """
    Full JSON-serialisable report of a completed simulation session.

    Exposed to agents and analysis tools. Contains NO internal vulnerability
    details — only what is physically measurable.
    """
    session_id:     str
    profile_name:   str
    config_summary: Dict[str, Any]
    metrics:        Dict[str, Any]
    observer_data:  Dict[str, Any]
    coherence:      Dict[str, Any]
    active_vulns:   List[str]       # IDs only — for ground-truth evaluation
    difficulty_score: float
    duration_s:     float
    n_iterations:   int

    def summary(self) -> str:
        lines = [
            f"{'─'*60}",
            f"  NuQKD Session Report  [{self.session_id}]",
            f"{'─'*60}",
            f"  Profile       : {self.profile_name}",
            f"  Difficulty    : {self.difficulty_score:.2f}",
            f"  Iterations    : {self.n_iterations}",
            f"  Duration      : {self.duration_s:.2f}s",
            f"",
            f"  ── Channel ──",
            f"  QBER (est)    : {self.metrics.get('qber_estimated', {}).get('mean', 'N/A')}",
            f"  QBER (actual) : {self.metrics.get('qber_actual', {}).get('mean', 'N/A')}",
            f"  Detection rate: {self.metrics.get('detection_rate', {}).get('mean', 'N/A')}",
            f"  Secret key    : {self.metrics.get('secret_key_bits', {}).get('mean', 'N/A')} bits/iter",
            f"  Throughput    : {self.metrics.get('throughput_bps', 'N/A')} bps",
            f"  Abort rate    : {self.metrics.get('abort_rate', 'N/A')}",
            f"",
            f"  ── Observer Highlights ──",
        ]
        for k, v in self.observer_data.items():
            if isinstance(v, float):
                lines.append(f"  {k:<30}: {v:.5g}")
            elif isinstance(v, bool):
                lines.append(f"  {k:<30}: {v}")
        lines += [
            f"",
            f"  ── Coherence Checks ──",
        ]
        for check, res in self.coherence.items():
            status = "✓" if res.get("passed") else "✗"
            lines.append(f"  {status} {check}: {res.get('comment','')}")
        lines += [
            f"",
            f"  ── Ground Truth (post-hoc only) ──",
            f"  Active vulnerabilities: {', '.join(self.active_vulns) or 'none'}",
            f"{'─'*60}",
        ]
        return "\n".join(lines)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps({
            "session_id":     self.session_id,
            "profile_name":   self.profile_name,
            "config_summary": self.config_summary,
            "metrics":        self.metrics,
            "observer_data":  self.observer_data,
            "coherence":      self.coherence,
            "active_vulns":   self.active_vulns,
            "difficulty_score": self.difficulty_score,
            "duration_s":     self.duration_s,
            "n_iterations":   self.n_iterations,
        }, indent=indent, default=str)


# ---------------------------------------------------------------------------
# QKDSession
# ---------------------------------------------------------------------------

class QKDSession:
    """
    Top-level simulation session.

    Instantiate once per scenario, call ``run()`` to execute all iterations.
    """

    def __init__(self,
                 config: SimulationConfig,
                 vuln_profile: Optional[VulnerabilityProfile] = None,
                 profile_name: str = "custom",
                 attack_agent=None,
                 seed: Optional[int] = None) -> None:
        self.config       = config
        self.vuln         = vuln_profile or VulnerabilityProfile()
        self.profile_name = profile_name
        self.attack       = attack_agent
        self.seed         = seed

        # Build protocol
        self.protocol = BB84Protocol(
            config       = config,
            vuln_profile = self.vuln,
            attack_agent = attack_agent,
            seed         = seed,
        )

        # Build observer (uses same vuln profile — but never exposes it directly)
        slot_ns = 1.0e9 / config.source.pulse_frequency_hz
        mu_declared = config.source.mean_photon_number
        T_approx    = 10 ** (-(
            config.channel.fiber.attenuation_db_per_km *
            config.channel.fiber.distance_km +
            config.channel.insertion_loss_db +
            config.detector.insertion_loss_db
        ) / 10.0)

        obs_rng = np.random.default_rng(
            None if seed is None else seed ^ 0xDEADBEEF
        )
        self.observer = ChannelObserver(
            channel_rng             = obs_rng,
            vulnerability_profile   = self.vuln,
            slot_duration_ns        = slot_ns,
            declared_mu             = mu_declared,
            declared_eta            = config.detector.efficiency,
            channel_transmittance   = T_approx,
        )

        self.metrics    = SessionMetrics()
        self._results:  List[BB84IterationResult] = []
        self._session_id = f"session_{int(time.time())}_{id(self) & 0xFFFF:04x}"

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_profile(cls,
                     profile_name: str = "medium",
                     config: Optional[SimulationConfig] = None,
                     attack_agent=None,
                     seed: Optional[int] = None) -> "QKDSession":
        """
        Convenience factory — create a session from a named profile.

        ``profile_name`` : one of "clean", "easy", "medium", "hard", "expert",
                           or "random".
        """
        if profile_name == "random":
            vuln = get_random_profile(seed)
            name = "random"
        elif profile_name in NAMED_PROFILES:
            vuln = NAMED_PROFILES[profile_name]
            name = profile_name
        else:
            raise ValueError(f"Unknown profile: {profile_name!r}. "
                             f"Choose from {list(NAMED_PROFILES)} + ['random']")

        cfg = config or SimulationConfig()
        return cls(cfg, vuln, name, attack_agent, seed)

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    def run(self,
            n_iterations: Optional[int] = None,
            run_observer: bool = True) -> SessionReport:
        """
        Execute the full session and return a ``SessionReport``.

        Parameters
        ----------
        n_iterations : int | None
            Override the config's ``num_iterations``.
        run_observer : bool
            If True, runs the full ChannelObserver measurement suite after
            the protocol iterations.  Set False to skip for speed.
        """
        t0 = time.perf_counter()
        n  = n_iterations or self.config.protocol.num_iterations

        logger.info("[Session %s] starting — profile=%s, n=%d, seed=%s",
                    self._session_id, self.profile_name, n, self.seed)

        # ── Run protocol iterations ────────────────────────────────────────
        self._results.clear()
        for i in range(n):
            result = self.protocol.run_iteration()
            self._results.append(result)
            self.metrics.ingest(result)

            if self.config.verbose:
                s = result.summary()
                logger.info("  iter %2d | det=%.3f | QBER=%-6s | key=%d | %s",
                            i, s["detection_rate"],
                            f"{s['qber']:.4f}" if s['qber'] is not None else "abort",
                            s["secret_key_bits"],
                            "ABORT" if s["aborted"] else "OK")

        # ── Run observer measurement suite ─────────────────────────────────
        obs_data: Dict[str, Any] = {}
        if run_observer:
            obs_data = self._run_observer_suite()

        # ── Coherence check ────────────────────────────────────────────────
        coherence = self.metrics.coherence_check(obs_data)

        # ── Build report ───────────────────────────────────────────────────
        duration = time.perf_counter() - t0

        active_vulns = [v.vuln_id for v in self.vuln.active_vulnerabilities()]

        report = SessionReport(
            session_id      = self._session_id,
            profile_name    = self.profile_name,
            config_summary  = self._config_summary(),
            metrics         = self.metrics.summary(),
            observer_data   = obs_data,
            coherence       = coherence,
            active_vulns    = active_vulns,
            difficulty_score = self.vuln.difficulty_score(),
            duration_s      = round(duration, 3),
            n_iterations    = n,
        )

        logger.info("[Session %s] done in %.2fs | vulns=%s",
                    self._session_id, duration, active_vulns)

        # Export if requested
        if self.config.export_results:
            self._export(report)

        return report

    # ------------------------------------------------------------------
    # Observer suite
    # ------------------------------------------------------------------

    def _run_observer_suite(self) -> Dict[str, Any]:
        """
        Run all ChannelObserver measurements and flatten key results
        into a single dict for the session report.
        """
        obs = self.observer
        out: Dict[str, Any] = {}

        # 1. Photon statistics
        phot = obs.measure_photon_statistics(n_pulses=5000)
        out.update({
            "mu_estimated":    phot.data["mu_estimated"],
            "mu_discrepancy":  phot.data["mu_discrepancy"],
            "p_multiphoton":   phot.data["p_multiphoton"],
            "pns_risk_score":  phot.data["pns_risk_score"],
        })

        # 2. Timing distribution
        timing = obs.sample_timing_distribution(n_samples=3000)
        out.update({
            "timing_bimodality_score": timing.data["bimodality_score"],
            "timing_skewness":         timing.data["skewness"],
            "timing_std_ns":           timing.data["std_ns"],
        })

        # 3. Conditional timing (basis-dependent)
        ctiming = obs.sample_basis_conditional_timing(n_per_basis=2000)
        out.update({
            "timing_delta_ns":       ctiming.data["delta_ns"],
            "timing_p_value":        ctiming.data["p_value"],
            "timing_significant":    ctiming.data["significant"],
        })

        # 4. Detection rate by intensity
        rates = obs.measure_detection_rate_by_intensity(n_pulses=6000)
        out.update({
            "signal_detection_rate": rates.data["signal_detection_rate"],
            "decoy_detection_rate":  rates.data["decoy_detection_rate"],
            "rate_ratio_anomaly":    rates.data["ratio_anomaly"],
            "decoy_identifiable":    rates.data["decoy_timing_identifiable"],
        })

        # 5. Detector efficiency
        det = obs.probe_detector_efficiency(n_probes=2000)
        out.update({
            "eta_detector_0":       det.data["eta_detector_0"],
            "eta_detector_1":       det.data["eta_detector_1"],
            "eta_mismatch_delta":   det.data["mismatch_delta_eta"],
            "mismatch_significant": det.data["mismatch_significant"],
            "time_shift_feasible":  det.data["time_shift_feasible"],
        })

        # 6. Dead time
        dead = obs.probe_dead_time(n_probes=200)
        out.update({
            "tau_det0_ns":  dead.data["tau_det0_ns"],
            "tau_det1_ns":  dead.data["tau_det1_ns"],
            "tau_delta_ns": dead.data["tau_delta_ns"],
        })

        # 7. Blinding threshold
        blind = obs.probe_blinding_threshold(max_photons=2000, n_steps=20)
        out.update({
            "blinding_detected":   blind.data["blinding_detected"],
            "blinding_threshold":  blind.data["blinding_threshold_est"],
        })

        # 8. Trojan Horse surface
        th = obs.probe_channel_reflectance()
        out.update({
            "max_reflectance":     th.data["max_reflectance"],
            "trojan_horse_risk":   th.data["trojan_horse_risk"],
            "peak_wavelength_nm":  th.data["peak_wavelength_nm"],
        })

        # 9. QBER baseline (from session metrics)
        qber_mean = self.metrics._safe_stat(self.metrics.qbers)["mean"]
        if not np.isnan(qber_mean):
            out["qber_baseline"] = qber_mean

        return out

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _config_summary(self) -> Dict[str, Any]:
        c = self.config
        return {
            "protocol":           c.protocol_name,
            "source_type":        c.source.type.value,
            "mu_declared":        c.source.mean_photon_number,
            "decoy_enabled":      c.source.enable_decoy,
            "channel_type":       c.channel.type.value,
            "distance_km":        c.channel.fiber.distance_km,
            "attenuation_db_km":  c.channel.fiber.attenuation_db_per_km,
            "depolarization":     c.channel.depolarization_prob,
            "detector_eta":       c.detector.efficiency,
            "dark_count_hz":      c.detector.dark_count_rate_hz,
            "dead_time_ns":       c.detector.dead_time_ns,
            "timing_jitter_ps":   c.detector.timing_jitter_ps,
            "raw_key_size":       c.protocol.raw_key_size,
            "qber_threshold":     c.protocol.qber_threshold,
            "ec_scheme":          c.protocol.ec_scheme.value,
            "n_iterations":       c.protocol.num_iterations,
        }

    def _export(self, report: SessionReport) -> None:
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{report.session_id}.json"
        path.write_text(report.to_json())
        logger.info("Report saved → %s", path)

    @property
    def results(self) -> List[BB84IterationResult]:
        """Access raw per-iteration results after ``run()``."""
        return list(self._results)
