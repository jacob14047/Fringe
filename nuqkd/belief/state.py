"""
nuqkd.belief.state
===================
Shared Bayesian belief state for the AI red-team.

This is the central nervous system of the QPET framework.  Every agent
reads from and writes to this object.  It stores:

  1. **Parameter estimates** — probability distributions over physical
     channel parameters (μ, η, τ, Δt, …).  Represented as (mean, std, confidence).

  2. **Anomaly scores** — per-vulnerability evidence accumulated from
     all measurements so far.

  3. **Attack hypotheses** — scored and ranked candidates for exploitation.

  4. **Observation history** — full log of all measurements taken.

  5. **QPET context** — current phase, iteration, budget remaining.

Thread safety
-------------
Not thread-safe by design — the orchestrator runs agents sequentially.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Parameter estimate
# ---------------------------------------------------------------------------

@dataclass
class ParameterEstimate:
    """
    Running estimate of a physical parameter with uncertainty.

    Stored as a Normal approximation to the posterior: (μ, σ, n_obs).
    Updated via online Bayesian update assuming Gaussian likelihood.
    """
    name:       str
    mean:       float
    std:        float
    n_obs:      int     = 0
    prior_mean: float   = 0.0
    prior_std:  float   = 1.0
    unit:       str     = ""
    bounds:     Tuple[float, float] = (-np.inf, np.inf)

    def update(self, new_value: float, measurement_std: float) -> None:
        """
        Bayesian update (conjugate Normal-Normal model).

        Prior: θ ~ N(μ_0, σ_0²)
        Likelihood: x | θ ~ N(θ, σ_meas²)
        Posterior: θ | x ~ N(μ_post, σ_post²)
        """
        sigma2_prior = self.std ** 2
        sigma2_meas  = measurement_std ** 2
        sigma2_post  = 1.0 / (1.0 / sigma2_prior + 1.0 / sigma2_meas)
        mu_post      = sigma2_post * (self.mean / sigma2_prior +
                                      new_value / sigma2_meas)
        self.mean    = float(np.clip(mu_post, self.bounds[0], self.bounds[1]))
        self.std     = float(np.sqrt(sigma2_post))
        self.n_obs  += 1

    def confidence_interval(self, alpha: float = 0.05) -> Tuple[float, float]:
        """Return (lower, upper) of the (1-α) credible interval."""
        z = scipy_stats.norm.ppf(1.0 - alpha / 2.0)
        lo = float(np.clip(self.mean - z * self.std, self.bounds[0], self.bounds[1]))
        hi = float(np.clip(self.mean + z * self.std, self.bounds[0], self.bounds[1]))
        return lo, hi

    def to_dict(self) -> Dict:
        return {
            "name": self.name, "mean": self.mean, "std": self.std,
            "n_obs": self.n_obs, "unit": self.unit,
            "ci_95": self.confidence_interval(0.05),
        }


# ---------------------------------------------------------------------------
# Attack hypothesis
# ---------------------------------------------------------------------------

class HypothesisStatus(str, Enum):
    PROPOSED   = "proposed"
    UNDER_TEST = "under_test"
    CONFIRMED  = "confirmed"
    REFUTED    = "refuted"
    EXECUTING  = "executing"
    SUCCESS    = "success"
    FAILED     = "failed"


@dataclass
class AttackHypothesis:
    """
    A candidate attack with its evidence base and execution parameters.
    """
    attack_id:    str
    attack_name:  str
    description:  str

    # Evidence
    prior_score:  float = 0.5    # From RAG retrieval
    evidence_score: float = 0.5  # Updated by measurements
    feasibility_score: float = 0.0  # From Attack Strategist

    # Constraints
    required_params:    Dict[str, Any] = field(default_factory=dict)
    estimated_qber:     float = 0.0
    detection_risk:     float = 0.0
    expected_info_gain: float = 0.0  # bits of key Eve learns

    # Status
    status:       HypothesisStatus = HypothesisStatus.PROPOSED
    attempts:     int   = 0
    best_result:  Optional[Dict] = None

    # Supporting evidence from measurements
    supporting_observations: List[str] = field(default_factory=list)
    rag_context:             str = ""

    @property
    def total_score(self) -> float:
        """Composite priority score (higher = more promising)."""
        return (0.35 * self.evidence_score +
                0.30 * self.prior_score +
                0.25 * self.feasibility_score +
                0.10 * (1.0 - self.detection_risk))

    def to_dict(self) -> Dict:
        return {
            "id":            self.attack_id,
            "name":          self.attack_name,
            "score":         round(self.total_score, 4),
            "evidence":      round(self.evidence_score, 4),
            "feasibility":   round(self.feasibility_score, 4),
            "estimated_qber": round(self.estimated_qber, 4),
            "detection_risk": round(self.detection_risk, 4),
            "info_gain_bits": round(self.expected_info_gain, 4),
            "status":        self.status.value,
            "attempts":      self.attempts,
        }


# ---------------------------------------------------------------------------
# Belief state
# ---------------------------------------------------------------------------

@dataclass
class BeliefState:
    """
    Central shared state for the QPET red-team.

    Agents do NOT modify this object directly from their constructor — they
    call the typed update methods to ensure consistency.
    """

    # ---- Channel parameter estimates -------------------------------------
    parameters: Dict[str, ParameterEstimate] = field(default_factory=dict)

    # ---- Anomaly evidence (per vulnerability ID) -------------------------
    # Values in [0, 1]: 0 = no evidence, 1 = strong evidence
    anomaly_scores: Dict[str, float] = field(default_factory=dict)

    # ---- Attack hypotheses (ranked by total_score) -----------------------
    hypotheses: List[AttackHypothesis] = field(default_factory=list)

    # ---- Observation log -------------------------------------------------
    observation_log: List[Dict[str, Any]] = field(default_factory=list)

    # ---- QPET session metadata -------------------------------------------
    session_id:       str   = ""
    phase:            str   = "init"    # init → profile → hypothesize → execute → done
    iteration:        int   = 0
    total_photons_used: int = 0
    total_detection_risk: float = 0.0

    # ---- LLM reasoning traces (for post-hoc analysis) -------------------
    reasoning_log: List[Dict[str, Any]] = field(default_factory=list)

    # ---- Ground truth (set by orchestrator; NEVER exposed to agents) ----
    _ground_truth: Optional[Dict] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._initialise_parameters()
        self._initialise_anomaly_scores()

    def _initialise_parameters(self) -> None:
        """Set up parameter estimates with uninformative priors."""
        defaults = [
            ParameterEstimate("mu_eff",           0.5,  0.3, unit="photons/pulse",
                              bounds=(0.001, 2.0)),
            ParameterEstimate("eta_detector",      0.5,  0.25, unit="",
                              bounds=(0.001, 1.0)),
            ParameterEstimate("eta_det0",          0.5,  0.25, unit="",
                              bounds=(0.001, 1.0)),
            ParameterEstimate("eta_det1",          0.5,  0.25, unit="",
                              bounds=(0.001, 1.0)),
            ParameterEstimate("dead_time_ns",      20000, 10000, unit="ns",
                              bounds=(0, 1e7)),
            ParameterEstimate("dead_time_det0_ns", 20000, 10000, unit="ns",
                              bounds=(0, 1e7)),
            ParameterEstimate("dead_time_det1_ns", 20000, 10000, unit="ns",
                              bounds=(0, 1e7)),
            ParameterEstimate("timing_delta_ps",   0.0,  30.0, unit="ps",
                              bounds=(-200, 200)),
            ParameterEstimate("channel_T",         0.1,  0.05, unit="",
                              bounds=(1e-6, 1.0)),
            ParameterEstimate("qber_baseline",     0.02, 0.02, unit="",
                              bounds=(0.0, 1.0)),
            ParameterEstimate("p_basis_z",         0.5,  0.05, unit="",
                              bounds=(0.0, 1.0)),
            ParameterEstimate("rng_period",        0.0,  500.0, unit="bits",
                              bounds=(0.0, 10000)),
            ParameterEstimate("blinding_threshold", 2000, 1000, unit="photons",
                              bounds=(0, 1e5)),
            ParameterEstimate("decoy_timing_offset_ps", 0.0, 20.0, unit="ps",
                              bounds=(-200, 200)),
        ]
        for p in defaults:
            self.parameters[p.name] = p

    def _initialise_anomaly_scores(self) -> None:
        vuln_ids = ["TC-01", "DM-01", "MU-01", "RNG-01",
                    "DT-01", "DB-01", "PA-01", "CASCADE-01"]
        for vid in vuln_ids:
            self.anomaly_scores[vid] = 0.0

    # ------------------------------------------------------------------
    # Update API (agents call these)
    # ------------------------------------------------------------------

    def update_parameter(self, name: str, value: float,
                          measurement_std: float) -> None:
        """Update a parameter estimate with a new observation."""
        if name not in self.parameters:
            self.parameters[name] = ParameterEstimate(name, value, measurement_std)
        else:
            self.parameters[name].update(value, measurement_std)

    def update_anomaly(self, vuln_id: str, delta: float,
                        source: str = "") -> None:
        """
        Increment an anomaly score.

        ``delta`` ∈ [-1, 1]: positive = more evidence, negative = counter-evidence.
        Score is clipped to [0, 1].
        """
        current = self.anomaly_scores.get(vuln_id, 0.0)
        # Weighted update: new evidence has diminishing returns as score → 1
        new = current + delta * (1.0 - current) if delta > 0 else current + delta
        self.anomaly_scores[vuln_id] = float(np.clip(new, 0.0, 1.0))

    def add_hypothesis(self, h: AttackHypothesis) -> None:
        """Add or update an attack hypothesis."""
        existing = next((x for x in self.hypotheses
                         if x.attack_id == h.attack_id), None)
        if existing is not None:
            # Merge: update scores
            existing.evidence_score    = h.evidence_score
            existing.feasibility_score = h.feasibility_score
            existing.required_params.update(h.required_params)
        else:
            self.hypotheses.append(h)
        self._rank_hypotheses()

    def _rank_hypotheses(self) -> None:
        self.hypotheses.sort(key=lambda h: h.total_score, reverse=True)

    def log_observation(self, agent_name: str,
                         measurement_type: str,
                         data: Dict[str, Any],
                         cost: int = 0) -> None:
        self.observation_log.append({
            "t":        time.time(),
            "agent":    agent_name,
            "type":     measurement_type,
            "data":     data,
            "cost":     cost,
        })
        self.total_photons_used += cost

    def log_reasoning(self, agent_name: str,
                       phase: str,
                       prompt: str,
                       response: str,
                       structured: Optional[Dict] = None) -> None:
        self.reasoning_log.append({
            "t":          time.time(),
            "agent":      agent_name,
            "phase":      phase,
            "prompt":     prompt[:500],   # truncate for storage
            "response":   response[:1000],
            "structured": structured,
        })

    # ------------------------------------------------------------------
    # Queries (agents call these to read state)
    # ------------------------------------------------------------------

    def get_parameter(self, name: str) -> Optional[ParameterEstimate]:
        return self.parameters.get(name)

    def top_hypotheses(self, n: int = 3) -> List[AttackHypothesis]:
        active = [h for h in self.hypotheses
                  if h.status not in (HypothesisStatus.REFUTED,
                                       HypothesisStatus.FAILED)]
        return active[:n]

    def top_anomalies(self, threshold: float = 0.3) -> Dict[str, float]:
        return {k: v for k, v in
                sorted(self.anomaly_scores.items(), key=lambda x: -x[1])
                if v >= threshold}

    def summary_dict(self) -> Dict[str, Any]:
        """Compact snapshot for LLM context window."""
        params = {
            name: {"mean": round(p.mean, 5), "std": round(p.std, 5), "n": p.n_obs}
            for name, p in self.parameters.items()
            if p.n_obs > 0
        }
        return {
            "phase":           self.phase,
            "iteration":       self.iteration,
            "photons_used":    self.total_photons_used,
            "parameters":      params,
            "anomaly_scores":  {k: round(v, 3) for k, v in
                                 self.anomaly_scores.items() if v > 0.05},
            "top_hypotheses":  [h.to_dict() for h in self.top_hypotheses(3)],
            "n_observations":  len(self.observation_log),
        }

    def to_json(self) -> str:
        d = self.summary_dict()
        return json.dumps(d, indent=2, default=str)
