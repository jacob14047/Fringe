"""
nuqkd.core.channel_observer
=============================
The *only* interface through which attack agents can observe the channel.

Design constraints
------------------
* The observer returns *statistical measurements*, never raw internal state.
* Every measurement costs a configurable number of photons / clock cycles.
* Active probing (e.g. Trojan Horse) consumes photons and *may* be detectable
  if Alice and Bob monitor their photon count statistics.
* The observer is physically faithful: measurement noise, finite-sample
  variance, and dead-time effects are all modelled.

Measurement taxonomy
--------------------
PASSIVE   — Eve observes photons in transit without additional disturbance.
            Cost: photons consumed but not forwarded → detectable via loss.
ACTIVE    — Eve injects probe photons or modifies the timing grid.
            Cost: additional photons on the channel → potentially detectable.
TIMING    — Eve measures arrival timestamps only; no state collapse.
            Cost: low; detection difficulty HIGH.
CLASSICAL — Eve reads the public classical channel (free, always available).

All measurements return a ``MeasurementResult`` containing the raw data plus
metadata (cost, detection risk, confidence).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from nuqkd.config.vulnerability_profiles import VulnerabilityProfile
from nuqkd.core.qubit import BASIS_X, BASIS_Z, Photon


# ---------------------------------------------------------------------------
# Measurement result container
# ---------------------------------------------------------------------------

class MeasurementType(str, Enum):
    PASSIVE  = "passive"
    ACTIVE   = "active"
    TIMING   = "timing"
    CLASSICAL = "classical"


@dataclass
class MeasurementResult:
    """
    Generic container for any observer measurement.

    ``data``        : dict with measurement-specific fields.
    ``cost_photons``: how many photons were consumed / disturbed.
    ``detection_risk``: estimated probability this measurement raises QBER
                        or loss statistics above threshold.
    ``confidence``  : statistical confidence in the result (0–1).
    ``n_samples``   : number of independent samples used.
    """
    measurement_type: MeasurementType
    data:             Dict[str, Any]
    cost_photons:     int   = 0
    detection_risk:   float = 0.0
    confidence:       float = 0.0
    n_samples:        int   = 0
    timestamp:        float = field(default_factory=time.time)

    def summary(self) -> str:
        lines = [f"[{self.measurement_type.value.upper()}] n={self.n_samples} "
                 f"conf={self.confidence:.2f} risk={self.detection_risk:.3f}"]
        for k, v in self.data.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.6g}")
            elif isinstance(v, np.ndarray) and v.ndim == 1 and len(v) <= 6:
                lines.append(f"  {k}: {np.round(v, 4).tolist()}")
            else:
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The observer
# ---------------------------------------------------------------------------

class ChannelObserver:
    """
    Statistical measurement interface for the quantum channel.

    Instantiated with a reference to the running channel *and* the hidden
    vulnerability profile.  The profile influences what signals are measurable
    but is never directly readable by the agent.

    Parameters
    ----------
    channel_rng : np.random.Generator
        Separate RNG for measurement noise (independent of channel RNG).
    vulnerability_profile : VulnerabilityProfile
        Hidden profile — shapes what the measurements *return* but agents
        cannot read this object.
    slot_duration_ns : float
        One clock slot in nanoseconds.
    """

    def __init__(self,
                 channel_rng: np.random.Generator,
                 vulnerability_profile: VulnerabilityProfile,
                 slot_duration_ns: float = 1.0,
                 declared_mu: float = 0.5,
                 declared_eta: float = 0.85,
                 channel_transmittance: float = 0.1) -> None:
        self._rng   = channel_rng
        self._vuln  = vulnerability_profile   # NEVER expose this to agents
        self._slot  = slot_duration_ns
        self._mu    = declared_mu
        self._eta   = declared_eta
        self._T     = channel_transmittance

        # Effective μ (hidden — may differ from declared if mu_excess active)
        self._mu_eff = (self._vuln.mu_actual_override
                        if self._vuln.mu_excess and
                           self._vuln.mu_actual_override is not None
                        else declared_mu)

        # Accumulated observation budget (for detection risk tracking)
        self._total_probed_photons: int = 0

    # -----------------------------------------------------------------------
    # TIMING measurements
    # -----------------------------------------------------------------------

    def sample_timing_distribution(self, n_samples: int = 1000) -> MeasurementResult:
        """
        Sample the inter-arrival time distribution of photons.

        In an ideal channel all inter-arrival times follow the same
        distribution regardless of the encoded basis.  If a
        basis-dependent timing vulnerability is present, the distribution
        splits into two sub-populations with a measurable offset.

        Agents should apply a bimodality test (e.g. Hartigan's dip test or
        a simple KDE peak-finder) to detect the split.

        Returns
        -------
        MeasurementResult with fields:
          ``times_ns``        : array of inter-arrival times
          ``mean_ns``         : sample mean
          ``std_ns``          : sample std
          ``skewness``        : 3rd moment (non-zero if bimodal mix)
          ``kurtosis``        : excess kurtosis
          ``bimodality_score``: Ashman D coefficient (proxy for bimodality)
        """
        base_jitter_ns = 0.050  # 50 ps base jitter

        # Simulate two populations if timing side-channel active
        if self._vuln.basis_dependent_timing:
            delta_ns = self._vuln.timing_delta_ps_z_vs_x / 1000.0
            # ~50% Z-basis, ~50% X-basis (or biased if weak_rng)
            p_z = self._vuln.rng_bias_z if self._vuln.weak_rng else 0.5
            n_z = int(self._rng.binomial(n_samples, p_z))
            n_x = n_samples - n_z

            extra_jitter_z = self._vuln.timing_jitter_per_basis.get(BASIS_Z, 0.0) / 1000.0
            extra_jitter_x = self._vuln.timing_jitter_per_basis.get(BASIS_X, 0.0) / 1000.0

            times_z = self._rng.normal(0.0,
                                        base_jitter_ns + extra_jitter_z,
                                        n_z)
            times_x = self._rng.normal(delta_ns,
                                        base_jitter_ns + extra_jitter_x,
                                        n_x)
            times = np.concatenate([times_z, times_x])
        else:
            times = self._rng.normal(0.0, base_jitter_ns, n_samples)

        self._rng.shuffle(times)

        # Ashman D coefficient (bimodality detection)
        mu1, mu2 = np.percentile(times, 25), np.percentile(times, 75)
        sigma = np.std(times) + 1e-12
        d = np.sqrt(2) * abs(mu1 - mu2) / sigma

        return MeasurementResult(
            measurement_type=MeasurementType.TIMING,
            data={
                "times_ns":        times,
                "mean_ns":         float(np.mean(times)),
                "std_ns":          float(np.std(times)),
                "skewness":        float(stats.skew(times)),
                "kurtosis":        float(stats.kurtosis(times)),
                "bimodality_score": float(d),
                "p25_ns":          float(np.percentile(times, 25)),
                "p75_ns":          float(np.percentile(times, 75)),
            },
            cost_photons=0,
            detection_risk=0.0,   # pure timing: zero extra loss
            confidence=1.0 - 1.0 / np.sqrt(n_samples),
            n_samples=n_samples,
        )

    def sample_basis_conditional_timing(self,
                                        n_per_basis: int = 2000) -> MeasurementResult:
        """
        Measure timing distribution *conditioned* on basis (requires Bob's
        cooperation or a separate measurement channel — this models Eve having
        partial timing access, e.g. via a fibre tap that preserves polarisation
        state).

        Returns separate timing statistics for Z and X basis slots.
        This is the key measurement for detecting the timing side-channel.
        """
        base_ns = 0.050
        delta_ns = (self._vuln.timing_delta_ps_z_vs_x / 1000.0
                    if self._vuln.basis_dependent_timing else 0.0)

        times_z = self._rng.normal(0.0,    base_ns, n_per_basis)
        times_x = self._rng.normal(delta_ns, base_ns, n_per_basis)

        # t-test for difference in means
        t_stat, p_val = stats.ttest_ind(times_z, times_x)

        return MeasurementResult(
            measurement_type=MeasurementType.TIMING,
            data={
                "mean_z_ns":       float(np.mean(times_z)),
                "mean_x_ns":       float(np.mean(times_x)),
                "delta_ns":        float(np.mean(times_x) - np.mean(times_z)),
                "t_statistic":     float(t_stat),
                "p_value":         float(p_val),
                "significant":     bool(p_val < 0.01),
                "effect_size_ns":  float(abs(np.mean(times_x) - np.mean(times_z))),
            },
            cost_photons=0,
            detection_risk=0.0,
            confidence=float(1.0 - p_val),
            n_samples=n_per_basis * 2,
        )

    # -----------------------------------------------------------------------
    # PHOTON STATISTICS measurements
    # -----------------------------------------------------------------------

    def measure_photon_statistics(self, n_pulses: int = 5000) -> MeasurementResult:
        """
        Measure the photon-number distribution of the source.

        If ``mu_excess`` is active, the measured mean will exceed the declared
        value — a key diagnostic for PNS vulnerability.

        Returns Poisson distribution fit parameters and goodness-of-fit.
        """
        counts = self._rng.poisson(self._mu_eff, n_pulses)

        # Fit Poisson distribution
        mu_hat = float(np.mean(counts))
        # Poisson goodness-of-fit via chi-squared
        max_k = min(int(mu_hat * 4) + 1, 15)
        observed_freq = np.array([np.sum(counts == k) for k in range(max_k + 1)])
        expected_freq = n_pulses * np.array(
            [np.exp(-mu_hat) * mu_hat**k / max(1, self._factorial(k))
             for k in range(max_k + 1)]
        )
        # Pool tail bins and normalise expected to match observed sum
        min_expected = 5
        mask = expected_freq >= min_expected
        if mask.sum() >= 2:
            obs_sel = observed_freq[mask].astype(float)
            exp_sel = expected_freq[mask].astype(float)
            # rescale expected to exact same total as observed (avoids scipy tolerance error)
            exp_sel = exp_sel * (obs_sel.sum() / exp_sel.sum())
            chi2, p_chi2 = stats.chisquare(obs_sel, exp_sel)
        else:
            chi2, p_chi2 = 0.0, 1.0

        p_vacuum    = float(np.mean(counts == 0))
        p_single    = float(np.mean(counts == 1))
        p_multi     = float(np.mean(counts >= 2))

        return MeasurementResult(
            measurement_type=MeasurementType.PASSIVE,
            data={
                "mu_estimated":      mu_hat,
                "mu_declared":       self._mu,
                "mu_discrepancy":    mu_hat - self._mu,
                "p_vacuum":          p_vacuum,
                "p_single_photon":   p_single,
                "p_multiphoton":     p_multi,
                "chi2_goodness":     float(chi2),
                "p_value_poisson":   float(p_chi2),
                "pns_risk_score":    float(p_multi / max(p_single, 1e-9)),
            },
            cost_photons=n_pulses,
            detection_risk=float(n_pulses / 1e6),   # small additional loss
            confidence=1.0 - 1.0 / np.sqrt(n_pulses),
            n_samples=n_pulses,
        )

    def measure_detection_rate_by_intensity(self,
                                             n_pulses: int = 10000) -> MeasurementResult:
        """
        Measure Bob's click rate for signal vs. decoy pulses separately.

        If ``decoy_timing_leak`` is active, Eve can identify decoy pulses from
        timing alone and compute separate detection rates — the difference
        reveals PNS attack feasibility.

        In an ideal channel, detection rates should scale with μ and be
        consistent with η · T.  Anomalies indicate PNS or channel manipulation.
        """
        # Signal detection rate
        eta_T    = self._eta * self._T
        rate_sig = 1.0 - np.exp(-self._mu_eff * eta_T)

        # Decoy (μ_d ~ 0.1 typically)
        mu_decoy     = 0.1
        rate_decoy   = 1.0 - np.exp(-mu_decoy * eta_T)

        # Add sampling noise
        n_sig   = n_pulses // 2
        n_decoy = n_pulses - n_sig
        clicks_sig   = self._rng.binomial(n_sig,   min(rate_sig, 1.0))
        clicks_decoy = self._rng.binomial(n_decoy, min(rate_decoy, 1.0))

        emp_rate_sig   = clicks_sig   / n_sig
        emp_rate_decoy = clicks_decoy / n_decoy

        # If decoy timing leak, Eve can identify decoy vs signal precisely
        decoy_identifiable = (self._vuln.decoy_timing_leak and
                              abs(self._vuln.decoy_timing_offset_ps) > 10.0)

        # Expected ratio under ideal channel: rate_sig / rate_decoy ≈ μ_s / μ_d
        expected_ratio = self._mu / max(mu_decoy, 1e-9)
        observed_ratio = emp_rate_sig / max(emp_rate_decoy, 1e-9)

        return MeasurementResult(
            measurement_type=MeasurementType.PASSIVE,
            data={
                "signal_detection_rate":     emp_rate_sig,
                "decoy_detection_rate":      emp_rate_decoy,
                "rate_ratio_observed":       observed_ratio,
                "rate_ratio_expected":       expected_ratio,
                "ratio_anomaly":             abs(observed_ratio - expected_ratio),
                "decoy_timing_identifiable": decoy_identifiable,
                "pns_detectable_by_decoy":   bool(abs(observed_ratio - expected_ratio) > 0.15),
            },
            cost_photons=n_pulses,
            detection_risk=float(n_pulses / 2e6),
            confidence=float(1.0 - 1.0 / np.sqrt(n_pulses)),
            n_samples=n_pulses,
        )

    # -----------------------------------------------------------------------
    # DETECTOR measurements
    # -----------------------------------------------------------------------

    def probe_detector_efficiency(self, n_probes: int = 3000) -> MeasurementResult:
        """
        Estimate detector efficiency by sending calibrated probe pulses.

        If ``detector_efficiency_mismatch`` is active, the two detectors
        (measuring |0⟩ and |1⟩) will show different efficiencies.
        This is the key precondition for the Time-Shift Attack.
        """
        eta_0 = self._eta
        eta_1 = self._eta + (self._vuln.eta_delta
                             if self._vuln.detector_efficiency_mismatch else 0.0)

        # Measure each detector separately with n_probes/2 pulses each
        n_each  = n_probes // 2
        clicks_0 = self._rng.binomial(n_each, eta_0)
        clicks_1 = self._rng.binomial(n_each, eta_1)

        eta_0_hat = clicks_0 / n_each
        eta_1_hat = clicks_1 / n_each
        mismatch  = eta_1_hat - eta_0_hat

        # Fisher's exact test for difference
        table = np.array([[clicks_0, n_each - clicks_0],
                          [clicks_1, n_each - clicks_1]])
        _, p_mismatch = stats.fisher_exact(table)

        # Time-shift attack feasibility: requires Δη > threshold
        time_shift_feasible = (self._vuln.detector_efficiency_mismatch and
                               abs(self._vuln.eta_delta) > 0.03)

        return MeasurementResult(
            measurement_type=MeasurementType.ACTIVE,
            data={
                "eta_detector_0":       eta_0_hat,
                "eta_detector_1":       eta_1_hat,
                "mismatch_delta_eta":   mismatch,
                "p_value_mismatch":     float(p_mismatch),
                "mismatch_significant": bool(p_mismatch < 0.05),
                "time_shift_feasible":  time_shift_feasible,
                "optimal_shift_ps":     self._vuln.exploit_window_offset_ps,
            },
            cost_photons=n_probes,
            detection_risk=0.02,   # slight increase in detected photons
            confidence=float(1.0 - p_mismatch),
            n_samples=n_probes,
        )

    def probe_dead_time(self, n_probes: int = 500) -> MeasurementResult:
        """
        Estimate detector dead time by sending closely-spaced double pulses.

        Dead time is the refractory period after a detection — no click is
        possible until it expires.  By varying the inter-pulse spacing and
        measuring the click probability of the second pulse, Eve can estimate τ.
        """
        tau_0 = 18_300.0  # ns — nominal (InGaAs SPAD default)
        tau_1 = tau_0 + (self._vuln.dead_time_delta_ns
                         if self._vuln.detector_efficiency_mismatch else 0.0)

        # Sweep inter-pulse spacing from 0 to 2τ
        spacings_ns = np.linspace(0, 2 * max(tau_0, tau_1), 50)
        rates_det0  = np.array([
            float(self._rng.binomial(20, float(t >= tau_0)) / 20)
            for t in spacings_ns
        ])
        rates_det1  = np.array([
            float(self._rng.binomial(20, float(t >= tau_1)) / 20)
            for t in spacings_ns
        ])

        # Estimate τ as the 50th percentile of the step function
        tau_0_est = float(spacings_ns[np.searchsorted(rates_det0, 0.5)])
        tau_1_est = float(spacings_ns[np.searchsorted(rates_det1, 0.5)])

        return MeasurementResult(
            measurement_type=MeasurementType.ACTIVE,
            data={
                "tau_det0_ns":        tau_0_est,
                "tau_det1_ns":        tau_1_est,
                "tau_delta_ns":       tau_1_est - tau_0_est,
                "sweep_spacings_ns":  spacings_ns,
                "click_rate_det0":    rates_det0,
                "click_rate_det1":    rates_det1,
            },
            cost_photons=n_probes,
            detection_risk=0.03,
            confidence=0.75,
            n_samples=n_probes,
        )

    def probe_blinding_threshold(self, max_photons: int = 2000,
                                  n_steps: int = 20) -> MeasurementResult:
        """
        Active probe: sweep probe pulse intensity to identify the detector
        blinding threshold.

        In Geiger mode the click probability is ~η_D regardless of intensity
        (for n ≥ 1).  In linear mode it scales with intensity.  The transition
        point reveals blinding feasibility.
        """
        intensities = np.linspace(1, max_photons, n_steps).astype(int)
        click_rates: List[float] = []

        for intensity in intensities:
            if (self._vuln.detector_blinding_vulnerable and
                    intensity > self._vuln.blinding_threshold_photons):
                # Linear mode: click rate proportional to intensity
                p_click = min(1.0, self._vuln.linear_mode_efficiency *
                              intensity / self._vuln.blinding_threshold_photons)
            else:
                # Geiger mode: saturated at η_D
                p_click = self._eta

            measured = float(self._rng.binomial(50, p_click) / 50)
            click_rates.append(measured)

        click_rates_arr = np.array(click_rates)
        # Detect transition point (biggest derivative)
        deriv     = np.abs(np.diff(click_rates_arr))
        trans_idx = int(np.argmax(deriv))
        blinding_threshold_est = float(intensities[trans_idx])

        return MeasurementResult(
            measurement_type=MeasurementType.ACTIVE,
            data={
                "intensities_photons":     intensities.tolist(),
                "click_rates":             click_rates_arr.tolist(),
                "blinding_threshold_est":  blinding_threshold_est,
                "blinding_detected":       bool(max(deriv) > 0.15),
                "linear_mode_slope":       float(
                    np.polyfit(intensities[trans_idx:],
                               click_rates_arr[trans_idx:], 1)[0]
                    if len(intensities[trans_idx:]) > 2 else 0.0
                ),
            },
            cost_photons=n_steps * 50,
            detection_risk=0.15,   # bright probes are very detectable
            confidence=0.85,
            n_samples=n_steps,
        )

    # -----------------------------------------------------------------------
    # RNG ANALYSIS
    # -----------------------------------------------------------------------

    def analyse_basis_sequence(self, sequence: np.ndarray) -> MeasurementResult:
        """
        Run NIST-inspired statistical tests on an observed basis sequence to
        detect weak / biased RNG.

        Tests performed:
        * Monobit frequency test (bias detection)
        * Runs test (autocorrelation)
        * Approximate entropy test
        * Periodicity test (FFT peak detection)

        A biased or periodic basis sequence allows Eve to predict future bases,
        enabling targeted intercept-resend attacks with reduced QBER footprint.
        """
        n = len(sequence)
        bits = sequence.astype(float)

        # 1. Monobit frequency test
        s_obs     = float(np.abs(np.sum(2 * bits - 1)) / np.sqrt(n))
        p_mono    = float(stats.norm.sf(s_obs) * 2)
        p_z_obs   = float(np.mean(bits))   # observed P(Z)

        # 2. Runs test
        runs = int(1 + np.sum(np.diff(bits) != 0))
        mu_runs = (2 * np.sum(bits == 0) * np.sum(bits == 1)) / n + 1
        if n > 1 and mu_runs > 0:
            sigma_runs = np.sqrt(
                (mu_runs - 1) * (mu_runs - 2) / max(n - 1, 1)
            )
            z_runs  = (runs - mu_runs) / max(sigma_runs, 1e-12)
            p_runs  = float(stats.norm.sf(abs(z_runs)) * 2)
        else:
            p_runs = 1.0

        # 3. Approximate entropy (m=2)
        def approx_entropy(x, m):
            n_ = len(x)
            phi = []
            for m_ in [m, m + 1]:
                templates = {}
                for i in range(n_ - m_ + 1):
                    key = tuple(x[i:i + m_].astype(int))
                    templates[key] = templates.get(key, 0) + 1
                total = n_ - m_ + 1
                phi.append(sum(
                    (c / total) * np.log(c / total + 1e-15)
                    for c in templates.values()
                ))
            return phi[0] - phi[1]

        apen = float(approx_entropy(bits, 2))

        # 4. Periodicity via FFT
        fft_mag  = np.abs(np.fft.rfft(bits - np.mean(bits)))
        fft_peak = float(np.max(fft_mag[1:]))   # exclude DC
        fft_mean = float(np.mean(fft_mag[1:]))
        peak_ratio = fft_peak / max(fft_mean, 1e-12)

        # Weak RNG if peak_ratio >> expected (~3 for random)
        period_detected = peak_ratio > 5.0
        if period_detected:
            peak_freq_idx = int(np.argmax(fft_mag[1:])) + 1
            period_est    = int(2 * (n // 2) / peak_freq_idx)
        else:
            period_est = 0

        return MeasurementResult(
            measurement_type=MeasurementType.CLASSICAL,
            data={
                "p_z_observed":      p_z_obs,
                "bias_delta":        abs(p_z_obs - 0.5),
                "p_value_monobit":   p_mono,
                "p_value_runs":      p_runs,
                "approx_entropy":    apen,
                "fft_peak_ratio":    peak_ratio,
                "period_detected":   period_detected,
                "period_estimate":   period_est,
                "rng_weak_verdict":  bool(p_mono < 0.01 or p_runs < 0.01 or
                                         period_detected),
            },
            cost_photons=0,
            detection_risk=0.0,
            confidence=float(1.0 - max(p_mono, p_runs)),
            n_samples=n,
        )

    # -----------------------------------------------------------------------
    # TROJAN HORSE surface
    # -----------------------------------------------------------------------

    def probe_channel_reflectance(self,
                                   wavelengths_nm: Optional[List[float]] = None
                                   ) -> MeasurementResult:
        """
        Active Trojan Horse probe: inject bright pulses at various wavelengths
        and measure backscattered intensity.

        In a real system, some wavelengths reflect off Alice's modulators and
        carry information about the modulator setting (= the basis).
        High reflectance at a specific λ indicates a Trojan Horse attack surface.
        """
        if wavelengths_nm is None:
            wavelengths_nm = [1310, 1480, 1530, 1550, 1570, 1600, 1625]

        # Simulate wavelength-dependent reflectance with a random resonance
        resonance_nm = 1530.0   # typical EOM resonance
        base_ref     = 0.001    # -30 dB baseline
        reflectances = []
        for wl in wavelengths_nm:
            # Lorentzian peak at resonance
            width = 20.0
            peak  = 0.08
            r = base_ref + peak * (width**2 / ((wl - resonance_nm)**2 + width**2))
            # Add measurement noise
            r += float(self._rng.normal(0, base_ref * 0.1))
            reflectances.append(max(0.0, r))

        max_ref = max(reflectances)
        peak_wl = wavelengths_nm[int(np.argmax(reflectances))]

        return MeasurementResult(
            measurement_type=MeasurementType.ACTIVE,
            data={
                "wavelengths_nm":     wavelengths_nm,
                "reflectances":       [round(r, 6) for r in reflectances],
                "max_reflectance":    max_ref,
                "peak_wavelength_nm": peak_wl,
                "trojan_horse_risk":  bool(max_ref > 0.01),
            },
            cost_photons=len(wavelengths_nm) * 1000,
            detection_risk=0.08,
            confidence=0.90,
            n_samples=len(wavelengths_nm),
        )

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    @staticmethod
    def _factorial(n: int) -> int:
        f = 1
        for i in range(2, n + 1):
            f *= i
        return f

    def observation_budget_report(self) -> Dict[str, Any]:
        return {
            "total_probed_photons": self._total_probed_photons,
            "estimated_loss_increase": self._total_probed_photons / 1e8,
        }
