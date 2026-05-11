"""
nuqkd.rag.knowledge_base
=========================
Quantum security knowledge base for RAG retrieval.

The knowledge base encodes attack prerequisites, detection signatures,
and exploitation techniques from the academic QKD security literature.

Each document has:
  * A unique ID
  * A natural-language description (used for embedding)
  * Structured metadata (used for filtering and scoring)
  * Physical prerequisites (parameters that must be true for the attack)
  * Detection signatures (what measurements reveal the vulnerability)

Documents are intentionally written to require *reasoning* to connect
to the channel's observed parameters — they are not lookup tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class KBDocument:
    """One entry in the knowledge base."""
    doc_id:       str
    title:        str
    content:      str          # Natural language — used for semantic search
    attack_class: str
    prerequisites: Dict[str, Any]   # {param_name: required_condition}
    signatures:   List[str]         # Observable signatures in measurements
    severity:     str               # "low" | "medium" | "high" | "critical"
    references:   List[str] = field(default_factory=list)
    tags:         List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Knowledge base documents
# ---------------------------------------------------------------------------

QUANTUM_ATTACK_KB: List[KBDocument] = [

    # -----------------------------------------------------------------------
    # PHOTON NUMBER SPLITTING
    # -----------------------------------------------------------------------
    KBDocument(
        doc_id="KB-PNS-01",
        title="Photon Number Splitting Attack on WCP Sources",
        content="""
The Photon Number Splitting (PNS) attack exploits multi-photon pulses from
weak coherent pulse (WCP) sources. When a laser pulse contains two or more
photons, an eavesdropper (Eve) can split off one photon and store it in quantum
memory without disturbing the remaining photons. She forwards the remaining
photons to Bob through a lossless channel. After basis announcement, Eve
measures her stored photon in the correct basis, obtaining full information.

The attack becomes feasible when the mean photon number mu is high
(typically mu > 0.3 with high channel loss). The multi-photon fraction
P(n>=2) = 1 - (1+mu)*exp(-mu) scales with mu squared for small mu.

For the attack to be undetected, Eve must preserve the expected detection
statistics. She replaces the lossy fiber with a lossless channel and blocks
single-photon pulses at the appropriate rate.

Key precondition: mu_actual must be high enough that P(n>=2) / P(n=1) gives
Eve sufficient information advantage. The threshold is approximately
mu_effective > 0.4 combined with channel transmittance T < 0.5.

If decoy states are employed, Eve cannot distinguish decoy from signal pulses.
Her blocking strategy applied uniformly across all intensities creates a
detectable anomaly: the vacancy rate of decoy pulses deviates from the
expected rate by a factor that scales with mu_signal / mu_decoy.
        """.strip(),
        attack_class="pns",
        prerequisites={
            "mu_effective":     "> 0.3",
            "source_type":      "wcp",
            "channel_loss_db":  "> 10",
        },
        signatures=[
            "mu_estimated significantly exceeds declared value",
            "p_multiphoton > 0.1",
            "vacancy_rate_signal >> vacancy_rate_decoy",
            "high pns_risk_score in photon statistics",
        ],
        severity="critical",
        references=["Brassard et al. PRL 85 2000", "GLLP PRA 2004"],
        tags=["pns", "wcp", "multi-photon", "quantum_memory"],
    ),

    # -----------------------------------------------------------------------
    # INTERCEPT AND RESEND
    # -----------------------------------------------------------------------
    KBDocument(
        doc_id="KB-IR-01",
        title="Intercept-and-Resend Attack",
        content="""
The intercept-and-resend (IR) attack is the simplest eavesdropping strategy.
Eve intercepts each photon from Alice, measures it in a randomly chosen basis
(Z or X), and re-prepares a new photon in the state she measured.

Mathematical impact: each intercepted photon has probability 1/2 of being
measured in the wrong basis. When this happens, Bob receives a random state
and gets the wrong bit with probability 1/2. Therefore:
  QBER contribution = epsilon / 4
where epsilon is the intercept fraction.

The attack is detectable because it increases QBER. For epsilon=1, QBER=25%,
well above the 11% BB84 security threshold. A careful Eve limits epsilon
to stay below the threshold:
  epsilon_max < 4 * qber_threshold = 44% for threshold=11%

This gives Eve at most 22% of the key bits, reducing security margin.

The optimal variant measures in the Breidbart basis (pi/8 rotation) to
maximize information while minimizing QBER perturbation.

This attack is most effective when: baseline QBER is already high (noise
masks the perturbation), the channel has high loss (fewer verification points),
or the QBER estimation sample is small (high statistical variance).
        """.strip(),
        attack_class="intercept_resend",
        prerequisites={
            "access_to_quantum_channel": True,
            "qber_threshold":            "> 0.05",
        },
        signatures=[
            "elevated QBER above baseline",
            "QBER variance consistent with epsilon/4 formula",
        ],
        severity="medium",
        references=["Bennett Brassard 1984", "Gisin et al RMP 2002"],
        tags=["ir", "intercept", "resend", "qber"],
    ),

    # -----------------------------------------------------------------------
    # TIMING SIDE CHANNEL
    # -----------------------------------------------------------------------
    KBDocument(
        doc_id="KB-TC-01",
        title="Basis-Dependent Timing Side-Channel Attack",
        content="""
In many practical QKD systems, the electro-optic modulator (EOM) used for
polarisation encoding introduces timing delays that differ between the two
bases. Z-basis encoding (0 and 90 degrees) and X-basis encoding (45 and
135 degrees) require different voltage levels and settling times in the EOM.

This creates a detectable timing signature: photons in different bases arrive
at slightly different times (delta_t, typically 10-100 picoseconds).

Eve can exploit this by measuring only the arrival time of photons (without
collapsing the quantum state if done carefully) and inferring the basis with
probability higher than 50%. This is a passive measurement — it introduces
no QBER increase.

Statistical detection method: collect the inter-arrival time distribution
of photons. If the distribution is bimodal (Ashman D coefficient >> 1) or
shows a significant Z-score when conditioned on basis announcements, the
side-channel is present.

With timing information, Eve can selectively intercept only photons in the
basis she has inferred, dramatically reducing the QBER contribution while
maintaining high information gain. The effective QBER contribution from a
timing-assisted IR attack is:
  QBER_timing = epsilon * P(wrong_basis_despite_timing) / 2
which can be near zero if delta_t >> timing_jitter.

Attack threshold: detectable with ~10,000 samples for delta_t = 20 ps,
or ~1,000 samples for delta_t = 50 ps.
        """.strip(),
        attack_class="timing_side_channel",
        prerequisites={
            "timing_delta_ps":  "> 15",
            "timing_jitter_ps": "< 3 * timing_delta_ps",
        },
        signatures=[
            "bimodality_score > 1.5 in timing distribution",
            "p_value < 0.01 in basis-conditional timing test",
            "significant effect_size_ns in conditional timing measurement",
            "skewness != 0 in arrival time distribution",
        ],
        severity="high",
        references=["Nauerth et al APL 2009", "Lamas-Linares PRA 2007"],
        tags=["timing", "side_channel", "eom", "passive"],
    ),

    # -----------------------------------------------------------------------
    # TIME-SHIFT ATTACK
    # -----------------------------------------------------------------------
    KBDocument(
        doc_id="KB-TSA-01",
        title="Time-Shift Attack Exploiting Detector Efficiency Mismatch",
        content="""
Bob's receiver typically contains two single-photon detectors (SPDs), one
for each eigenstate in the measurement basis. Due to manufacturing tolerances,
these detectors have slightly different quantum efficiencies eta_0 and eta_1.

The time-shift attack exploits this mismatch: Eve shifts the arrival time of
each photon to fall within the optimal detection window of the detector
corresponding to the bit value she wants Bob to register.

If detector 0 (for bit=0) has efficiency eta_0 and detector 1 (for bit=1)
has efficiency eta_1 = eta_0 + delta_eta, then by shifting photon arrival
times, Eve increases the probability of a particular outcome.

The attack introduces NO additional QBER if the shift is chosen optimally:
  optimal_shift = argmax_t [ eta_0(t) / eta_1(t) ]
This is the time offset where the ratio of detection efficiencies is maximized.

Required conditions:
1. delta_eta = |eta_1 - eta_0| > 0.03 (measurable mismatch)
2. The detector gating window must have a time-varying efficiency profile
3. Dead time delta must be > 500 ns (common in InGaAs SPADs)

Detection by Alice and Bob: very difficult. The attack preserves QBER and
photon statistics. Only detectable by hardware characterization of the
detector efficiency vs. time profile.

Measurement approach: probe each detector separately with calibrated pulses
at different arrival times. Fisher's exact test on click counts reveals
the mismatch. A ratio |eta_1 - eta_0| / eta_mean > 0.05 suggests
time-shift attack feasibility.
        """.strip(),
        attack_class="time_shift",
        prerequisites={
            "eta_mismatch": "> 0.03",
            "detector_gated": True,
            "dead_time_ns": "> 500",
        },
        signatures=[
            "mismatch_significant=True in detector probe",
            "mismatch_delta_eta > 0.03",
            "tau_delta_ns significantly > 0",
            "p_value_mismatch < 0.05 in Fisher test",
        ],
        severity="high",
        references=["Qi et al J.Cryptology 2007", "Zhao et al PRA 2008"],
        tags=["time_shift", "detector", "mismatch", "gating"],
    ),

    # -----------------------------------------------------------------------
    # DETECTOR BLINDING
    # -----------------------------------------------------------------------
    KBDocument(
        doc_id="KB-DB-01",
        title="Detector Blinding Attack (Makarov Attack)",
        content="""
The detector blinding attack, demonstrated experimentally by Makarov et al.
in 2010, completely breaks the security of commercial QKD systems without
being detected.

Physical basis: InGaAs SPADs normally operate in Geiger mode (biased above
breakdown voltage) where a single photon triggers an avalanche. If flooded
with bright continuous light (typically > 10^6 photons), the SPAD quenches
and transitions to linear mode, where it behaves as a classical photodiode.

In linear mode:
- No single-photon sensitivity
- Click only when pulse intensity exceeds a threshold
- Eve controls which detector clicks by controlling pulse intensity

Attack sequence:
1. Eve blinds both of Bob's detectors with CW illumination
2. Eve intercepts Alice's photons and measures them
3. Eve sends bright classical pulses to Bob: only the detector corresponding
   to Eve's measured bit value receives a pulse above threshold
4. Bob's detectors click deterministically according to Eve's control
5. QBER = 0, statistics unchanged, attack is invisible

Key diagnostic: the transition from Geiger to linear mode is detectable
by sweeping probe intensity and observing the click-rate vs. intensity curve.
In Geiger mode: click rate saturates at eta_D for n >= 1.
In linear mode: click rate scales linearly with intensity.

The transition knee point gives the blinding threshold.
A blinding_threshold < 5000 photons/pulse combined with linear_mode_efficiency
> 0.8 indicates blinding vulnerability.
        """.strip(),
        attack_class="detector_blinding",
        prerequisites={
            "blinding_threshold_photons": "< 5000",
            "detector_type":              "InGaAs_SPAD",
            "linear_mode_efficiency":     "> 0.5",
        },
        signatures=[
            "blinding_detected=True in blinding probe",
            "click_rate increases linearly above threshold intensity",
            "blinding_threshold_est < 5000",
            "linear_mode_slope > 0 in intensity sweep",
        ],
        severity="critical",
        references=["Makarov et al NJP 2009", "Lydersen et al Nature Photon 2010"],
        tags=["blinding", "detector", "makarov", "linear_mode", "spad"],
    ),

    # -----------------------------------------------------------------------
    # TROJAN HORSE
    # -----------------------------------------------------------------------
    KBDocument(
        doc_id="KB-TH-01",
        title="Trojan Horse Attack via Back-Reflected Probe Photons",
        content="""
The Trojan Horse attack (THA) allows Eve to learn Alice's basis and bit
choices WITHOUT intercepting the transmitted photons, leaving zero QBER.

Attack mechanism:
1. Eve injects bright probe pulses into Alice's quantum channel input port
2. These pulses propagate backwards through Alice's optical path
3. Alice's electro-optic modulators (EOM) partially reflect the probe
4. The reflected light carries spectral/polarisation information about the
   current modulator setting (= Alice's basis and bit)
5. Eve intercepts the reflected pulses and reads Alice's settings

Key condition: Alice's optical isolator must have finite isolation
(typical: 30-40 dB). At a probe power that saturates Alice's detector
but doesn't trigger the isolator alarm, this attack works.

Spectral analysis: the reflected spectrum depends on the EOM resonance
frequency. By sweeping probe wavelengths and measuring reflectance,
Eve identifies the resonant wavelength with maximum information leakage.

Observable: The channel_reflectance measurement shows wavelength-dependent
backscatter with a Lorentzian peak at the EOM resonance. If max_reflectance
> 0.01 (-20 dB), the Trojan Horse surface is exploitable.

Countermeasure: wavelength filters at Alice's port, optical power monitors.
If not implemented, any wavelength with reflectance > 0.005 is a potential
THA channel.
        """.strip(),
        attack_class="trojan_horse",
        prerequisites={
            "max_reflectance": "> 0.01",
            "optical_isolator_isolation_db": "< 40",
        },
        signatures=[
            "trojan_horse_risk=True in reflectance probe",
            "max_reflectance > 0.01",
            "clear peak in reflectance vs. wavelength",
        ],
        severity="critical",
        references=["Gisin et al QIC 2006", "Jain et al PRL 2014"],
        tags=["trojan_horse", "back_reflection", "eom", "alice_side"],
    ),

    # -----------------------------------------------------------------------
    # WEAK RNG / BIASED BASIS
    # -----------------------------------------------------------------------
    KBDocument(
        doc_id="KB-RNG-01",
        title="Exploiting Biased or Periodic Basis Generation",
        content="""
QKD security proofs assume Alice's basis choices are perfectly random and
independent. If the random number generator (RNG) is biased or periodic,
Eve can exploit the predictability.

Bias attack: if P(Z-basis) = p_z != 0.5, then for a fraction |p_z - 0.5|
of qubits Eve can guess the correct basis without measuring. This reduces
the effective security parameter.

Periodicity attack: if the basis sequence has period T (detectable via FFT),
Eve can predict the basis for all future qubits after observing T initial
ones. This is the most severe RNG vulnerability.

Detection method (applied to classical channel observation of basis announcements):
1. Monobit test: |mean(bases) - 0.5| > 0.01 indicates bias (p < 0.01)
2. Runs test: autocorrelation in the sequence (p_value < 0.01)
3. FFT periodicity: peak-to-mean ratio > 5 in the frequency domain
4. Approximate entropy: low ApEn (< 0.7) indicates low randomness

If period_detected=True with period_estimate T:
- Eve waits T + 1 basis announcements, identifies period
- For subsequent rounds she knows every basis with certainty
- IR attack with known bases: QBER = 0

If bias_delta > 0.05:
- Eve always guesses the more frequent basis
- Information advantage = bias_delta fraction of sifted bits
        """.strip(),
        attack_class="rng_exploit",
        prerequisites={
            "rng_bias_delta": "> 0.03",
            "rng_periodic_or_biased": True,
        },
        signatures=[
            "p_value_monobit < 0.01",
            "rng_weak_verdict=True",
            "period_detected=True in basis analysis",
            "fft_peak_ratio > 5",
            "bias_delta > 0.05",
        ],
        severity="high",
        references=["NIST SP800-22", "Ma et al PRA 2005"],
        tags=["rng", "bias", "periodicity", "classical", "basis_prediction"],
    ),

    # -----------------------------------------------------------------------
    # DECOY STATE IDENTIFICATION
    # -----------------------------------------------------------------------
    KBDocument(
        doc_id="KB-DT-01",
        title="Decoy State Identification via Timing Correlation",
        content="""
Decoy-state QKD relies on Eve's inability to distinguish signal pulses from
decoy pulses. If a timing correlation exists between the pulse type and the
emission time, this security assumption is violated.

Physical mechanism: Alice uses two (or more) lasers with different mean
photon numbers. The second laser (decoy source) typically has different
cavity characteristics — slightly different settling time, turn-on jitter,
or repetition timing. This creates a timing offset (delta_t_decoy) between
signal and decoy pulses.

If delta_t_decoy > 2 * sigma_jitter, Eve can identify decoy pulses by
measuring only their arrival time, without touching the quantum state.

Consequence: Eve applies PNS selectively to signal pulses only.
The decoy pulse statistics remain unperturbed, so the decoy detection
test passes even though PNS is active on signal pulses.

Measurement: The detect_detection_rate_by_intensity measurement includes
a flag decoy_timing_identifiable. If True, the security of the entire
decoy protocol is compromised.

Diagnostic signature: run timing measurements separately during periods
when decoy pulses are expected (identified from classical channel) vs.
signal periods. A t-test on the timing distributions with p < 0.001
confirms the timing leakage.
        """.strip(),
        attack_class="decoy_bypass",
        prerequisites={
            "decoy_timing_offset_ps": "> 15",
            "decoy_enabled": True,
        },
        signatures=[
            "decoy_timing_identifiable=True",
            "decoy_timing_offset significant in timing analysis",
        ],
        severity="high",
        references=["Lo et al PRL 2005 (decoy protocol)", "Zhao et al QIC 2009"],
        tags=["decoy", "timing", "pns", "identification"],
    ),

    # -----------------------------------------------------------------------
    # PRIVACY AMPLIFICATION WEAKNESS
    # -----------------------------------------------------------------------
    KBDocument(
        doc_id="KB-PA-01",
        title="Privacy Amplification with Low-Entropy Seed",
        content="""
Privacy amplification (PA) uses a universal hash function to compress the
reconciled key, removing Eve's information. The security of PA depends on
the hash function being chosen uniformly at random from the family.

If the PA seed (which selects the hash function) has insufficient entropy,
Eve can narrow the hash family and potentially invert the compression.

Attack conditions:
1. Seed entropy < 128 bits: Eve can enumerate seed candidates
2. Seed derived from public data (timestamps, sequence hashes): predictable
3. Seed reuse across sessions: Eve accumulates information across multiple
   key exchanges

Information leakage: if Eve knows k bits of a 256-bit PA seed, she can
reduce the search space by 2^k. For k = 128, a classical computer can
invert the hash for keys up to ~256 bits long.

This attack is passive — no quantum channel interaction needed. Eve
observes only the classical channel and performs offline computation.

Detection: very difficult to detect from Alice/Bob's perspective. The QBER
and all quantum statistics remain normal.

Diagnostic: analyze the PA protocol parameters (seed generation method,
entropy source). If pa_seed_entropy_bits < 128, the vulnerability is active.
        """.strip(),
        attack_class="pa_weakness",
        prerequisites={
            "pa_seed_entropy_bits": "< 128",
        },
        signatures=[
            "pa_seed_entropy_bits < 128 (requires classical analysis)",
            "seed derived from predictable sources",
        ],
        severity="critical",
        references=["Bennett et al J.Cryptology 1995", "Tomamichel et al IEEE 2011"],
        tags=["privacy_amplification", "hash", "seed", "entropy", "offline"],
    ),

    # -----------------------------------------------------------------------
    # GENERAL QKD PARAMETER BOUNDS
    # -----------------------------------------------------------------------
    KBDocument(
        doc_id="KB-GEN-01",
        title="BB84 Security Thresholds and Attack Feasibility",
        content="""
BB84 security guarantees hold under specific parameter constraints.
Understanding these thresholds is essential for determining which attacks
are feasible given observed parameters.

QBER threshold for unconditional security:
- Individual attacks (no quantum memory): QBER < 11%
- Collective attacks: QBER < 11% (same bound with Devetak-Winter)
- Coherent attacks: QBER < 11% (Shor-Preskill security proof)

If QBER_baseline is already close to 11%, there is little room for
Eve to inject additional errors without being detected. Attacks that
introduce delta_QBER should satisfy:
  QBER_baseline + delta_QBER < qber_threshold

Secret key rate (asymptotic): r = 1 - (1 + f_EC) * H2(QBER)
- r = 0 when QBER > 11%
- r is maximized at QBER = 0 (ideal channel)

Key parameters affecting attack strategy selection:
1. mu_eff: high mu favors PNS over IR
2. channel_T: low T favors PNS (Eve can replace with lossless channel)
3. eta_detector: determines Bob's expected click rate
4. QBER_baseline: determines headroom for error-introducing attacks
5. raw_key_size N: small N gives high statistical uncertainty in QBER estimate
   (fluctuations ~ 1/sqrt(N*f_sharing)) allowing attacks below detection threshold
        """.strip(),
        attack_class="general",
        prerequisites={},
        signatures=["all channels"],
        severity="informational",
        references=["Shor Preskill PRL 2000", "Devetak Winter PRL 2005"],
        tags=["thresholds", "security", "qber", "key_rate"],
    ),
]


def get_all_documents() -> List[KBDocument]:
    return list(QUANTUM_ATTACK_KB)


def get_document_by_id(doc_id: str) -> Optional[KBDocument]:
    return next((d for d in QUANTUM_ATTACK_KB if d.doc_id == doc_id), None)
