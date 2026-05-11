"""
nuqkd.metrics.collector
========================
Aggregates per-iteration results into session-level statistics.

Provides:
  * Running means / std for QBER, key rate, detection rate
  * Click-type breakdown (genuine / dark / afterpulse)
  * Abort rate per session
  * Observer-coherence check: does ChannelObserver data match ground truth?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class SessionMetrics:
    """Aggregated statistics for a full BB84 session."""

    n_iterations:     int   = 0
    n_aborted:        int   = 0

    # Detection
    detection_rates:   List[float] = field(default_factory=list)
    # QBER
    qbers:             List[float] = field(default_factory=list)
    actual_qbers:      List[float] = field(default_factory=list)
    # Key material
    sifted_lengths:    List[int]   = field(default_factory=list)
    secret_key_lengths: List[int]  = field(default_factory=list)
    # Timing
    wall_times:        List[float] = field(default_factory=list)
    # Channel
    transmittances:    List[float] = field(default_factory=list)
    photons_survived:  List[int]   = field(default_factory=list)

    # Click type counters (summed across all iterations)
    genuine_clicks:    int = 0
    dark_clicks:       int = 0
    afterpulse_clicks: int = 0

    def ingest(self, result) -> None:
        """Add one BB84IterationResult to the aggregate."""
        from nuqkd.protocols.bb84 import BB84IterationResult
        r: BB84IterationResult = result

        self.n_iterations += 1

        if r.aborted:
            self.n_aborted += 1
            return

        self.detection_rates.append(r.detection_rate)
        self.qbers.append(r.qber)
        self.transmittances.append(r.channel_T)
        self.photons_survived.append(r.photons_survived)
        self.wall_times.append(r.wall_time_s)

        if r.postproc is not None:
            self.actual_qbers.append(r.postproc.qber.actual_qber)
            self.sifted_lengths.append(r.n_sifted)
            self.secret_key_lengths.append(r.secret_key_bits)

        # Click breakdown
        if len(r.click_types) > 0:
            self.genuine_clicks    += int(np.sum(r.click_types == "genuine"))
            self.dark_clicks       += int(np.sum(r.click_types == "dark"))
            self.afterpulse_clicks += int(np.sum(r.click_types == "afterpulse"))

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def _safe_stat(self, lst: List[float]) -> Dict[str, float]:
        if not lst:
            return {"mean": float("nan"), "std": float("nan"),
                    "min": float("nan"), "max": float("nan")}
        a = np.array(lst)
        return {
            "mean": round(float(np.mean(a)), 6),
            "std":  round(float(np.std(a)),  6),
            "min":  round(float(np.min(a)),  6),
            "max":  round(float(np.max(a)),  6),
        }

    def summary(self) -> Dict[str, Any]:
        n_ok = self.n_iterations - self.n_aborted
        total_clicks = (self.genuine_clicks + self.dark_clicks +
                        self.afterpulse_clicks)

        return {
            "iterations":        self.n_iterations,
            "aborted":           self.n_aborted,
            "abort_rate":        round(self.n_aborted / max(self.n_iterations, 1), 4),
            "detection_rate":    self._safe_stat(self.detection_rates),
            "qber_estimated":    self._safe_stat(self.qbers),
            "qber_actual":       self._safe_stat(self.actual_qbers),
            "sifted_length":     self._safe_stat(self.sifted_lengths),
            "secret_key_bits":   self._safe_stat(self.secret_key_lengths),
            "transmittance":     self._safe_stat(self.transmittances),
            "wall_time_s":       self._safe_stat(self.wall_times),
            "click_breakdown": {
                "genuine":    self.genuine_clicks,
                "dark":       self.dark_clicks,
                "afterpulse": self.afterpulse_clicks,
                "total":      total_clicks,
                "dark_fraction": round(
                    self.dark_clicks / max(total_clicks, 1), 4
                ),
            },
            "throughput_bps": round(
                sum(self.secret_key_lengths) / max(sum(self.wall_times), 1e-9), 2
            ),
        }

    def coherence_check(self, observer_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify that ChannelObserver measurements are statistically consistent
        with ground-truth session metrics.

        Returns a dict with one entry per check:
            {"check_name": {"passed": bool, "delta": float, "comment": str}}
        """
        checks = {}

        # 1. μ estimate vs actual detection rate
        mu_est = observer_results.get("mu_estimated")
        T_mean = self._safe_stat(self.transmittances)["mean"]
        eta    = 0.85   # detector efficiency (from default config)
        if mu_est is not None and not np.isnan(T_mean):
            expected_rate = 1.0 - np.exp(-mu_est * T_mean * eta)
            actual_rate   = self._safe_stat(self.detection_rates)["mean"]
            delta         = abs(expected_rate - actual_rate)
            checks["mu_detection_rate_coherence"] = {
                "passed":  delta < 0.05,
                "expected_rate": round(float(expected_rate), 4),
                "actual_rate":   round(float(actual_rate), 4),
                "delta":         round(float(delta), 4),
                "comment": ("OK — μ estimate consistent with detection rate"
                            if delta < 0.05 else
                            "ANOMALY — μ estimate inconsistent with detection rate"),
            }

        # 2. QBER baseline consistency
        qber_obs  = observer_results.get("qber_baseline")
        qber_sess = self._safe_stat(self.qbers)["mean"]
        if qber_obs is not None and not np.isnan(qber_sess):
            delta = abs(qber_obs - qber_sess)
            checks["qber_coherence"] = {
                "passed":   delta < 0.03,
                "observer": round(float(qber_obs), 4),
                "session":  round(float(qber_sess), 4),
                "delta":    round(float(delta), 4),
                "comment":  ("OK" if delta < 0.03 else
                             "ANOMALY — QBER estimate mismatch"),
            }

        return checks
