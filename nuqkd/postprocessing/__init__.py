"""
nuqkd.postprocessing
=====================
Classical post-processing pipeline for QKD.

Stages
------
1. **Sifting** — keep only slots where Alice and Bob used the same basis.
2. **QBER estimation** — publicly compare a fraction of the sifted key to
   estimate the error rate.
3. **Error correction** — reconcile the remaining sifted bits.
4. **Privacy amplification** — compress the reconciled key to remove any
   information Eve might have obtained.

Each stage is exposed as a standalone function to allow protocol designers
to substitute alternative algorithms (e.g., a different PA hash).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from nuqkd.config.parameters import ErrorCorrectionScheme, ProtocolConfig


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class SiftingResult:
    sifted_alice:   np.ndarray   = field(default_factory=lambda: np.array([], dtype=int))
    sifted_bob:     np.ndarray   = field(default_factory=lambda: np.array([], dtype=int))
    sifted_indices: np.ndarray   = field(default_factory=lambda: np.array([], dtype=int))
    sifting_efficiency: float    = 0.0   # n_sifted / n_raw


@dataclass
class QBERResult:
    estimated_qber: float        = 0.0
    actual_qber:    float        = 0.0   # available in simulation (not in real QKD)
    n_shared:       int          = 0
    n_errors_found: int          = 0
    aborted:        bool         = False
    # Remaining (non-shared) bits for key generation
    remaining_alice: np.ndarray  = field(default_factory=lambda: np.array([], dtype=int))
    remaining_bob:   np.ndarray  = field(default_factory=lambda: np.array([], dtype=int))
    remaining_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))


@dataclass
class ECResult:
    corrected_bob:  np.ndarray   = field(default_factory=lambda: np.array([], dtype=int))
    parity_bits_revealed: int    = 0     # bits leaked to Eve during EC
    success: bool                = True


@dataclass
class PAResult:
    secret_key:    np.ndarray    = field(default_factory=lambda: np.array([], dtype=int))
    secret_key_bits: int         = 0
    pa_compression: float        = 0.0  # output / input length ratio


@dataclass
class PostProcessingResult:
    sifting:        SiftingResult  = field(default_factory=SiftingResult)
    qber:           QBERResult     = field(default_factory=QBERResult)
    ec:             ECResult       = field(default_factory=ECResult)
    pa:             PAResult       = field(default_factory=PAResult)
    secret_key:     np.ndarray     = field(default_factory=lambda: np.array([], dtype=int))
    secret_key_rate: float         = 0.0   # bits per raw photon
    aborted:        bool           = False
    abort_reason:   str            = ""


# ---------------------------------------------------------------------------
# 1. Sifting
# ---------------------------------------------------------------------------

def sift(alice_bits: np.ndarray,
         alice_bases: np.ndarray,
         bob_bits: np.ndarray,
         bob_bases: np.ndarray,
         detected_mask: np.ndarray) -> SiftingResult:
    """
    Sifting step: keep only positions where (a) Bob detected a photon and
    (b) Alice and Bob used the same basis.

    Parameters
    ----------
    alice_bits, alice_bases : ndarray of int, shape (N,)
    bob_bits, bob_bases     : ndarray of int, shape (N,)
        Only defined for slots where Bob detected (use ``detected_mask`` to
        filter; undetected entries can be anything).
    detected_mask : ndarray of bool, shape (N,)
        True for slots where Bob had a detection event.

    Returns
    -------
    SiftingResult
    """
    # First filter to detected slots
    det_indices = np.where(detected_mask)[0]
    a_bits_det  = alice_bits[det_indices]
    a_bases_det = alice_bases[det_indices]
    b_bits_det  = bob_bits[det_indices]
    b_bases_det = bob_bases[det_indices]

    # Then keep only matching bases
    match_mask   = (a_bases_det == b_bases_det)
    sifted_idx   = det_indices[match_mask]

    result = SiftingResult()
    result.sifted_alice       = a_bits_det[match_mask]
    result.sifted_bob         = b_bits_det[match_mask]
    result.sifted_indices     = sifted_idx
    result.sifting_efficiency = len(sifted_idx) / max(len(alice_bits), 1)
    return result


# ---------------------------------------------------------------------------
# 2. QBER estimation
# ---------------------------------------------------------------------------

def estimate_qber(sifted_alice: np.ndarray,
                  sifted_bob:   np.ndarray,
                  sifted_indices: np.ndarray,
                  sharing_rate: float,
                  qber_threshold: float,
                  rng: np.random.Generator) -> QBERResult:
    """
    Publicly compare a fraction ``sharing_rate`` of the sifted key bits to
    estimate the QBER.  Shared bits are discarded afterwards.

    The error rate of the *shared* bits is used as an estimator for the
    error rate on the *remaining* bits::

        QBER_est ≈ n_errors_shared / n_shared

    Parameters
    ----------
    sifted_alice, sifted_bob : ndarray of int
        Sifted key bits from each party.
    sifted_indices : ndarray of int
        Original pulse indices corresponding to the sifted positions.
    sharing_rate : float
        Fraction of sifted key exposed publicly (0 < f < 1).
    qber_threshold : float
        Abort threshold.
    rng : np.random.Generator

    Returns
    -------
    QBERResult
    """
    n_sifted  = len(sifted_alice)
    n_shared  = max(1, int(np.round(n_sifted * sharing_rate)))
    n_shared  = min(n_shared, n_sifted)

    # Choose shared positions uniformly at random (without replacement)
    shared_pos = rng.choice(n_sifted, size=n_shared, replace=False)
    shared_pos = np.sort(shared_pos)
    remain_pos = np.setdiff1d(np.arange(n_sifted), shared_pos)

    shared_alice = sifted_alice[shared_pos]
    shared_bob   = sifted_bob[shared_pos]
    n_errors     = int(np.sum(shared_alice != shared_bob))
    qber_est     = n_errors / n_shared

    # True QBER (simulation only — not available in real QKD)
    true_errors  = int(np.sum(sifted_alice != sifted_bob))
    qber_actual  = true_errors / n_sifted if n_sifted > 0 else 0.0

    result = QBERResult()
    result.estimated_qber   = qber_est
    result.actual_qber      = qber_actual
    result.n_shared         = n_shared
    result.n_errors_found   = n_errors
    result.aborted          = qber_est > qber_threshold
    result.remaining_alice  = sifted_alice[remain_pos]
    result.remaining_bob    = sifted_bob[remain_pos]
    result.remaining_indices= sifted_indices[remain_pos]
    return result


# ---------------------------------------------------------------------------
# 3. Error correction
# ---------------------------------------------------------------------------

def error_correct(alice_bits: np.ndarray,
                  bob_bits:   np.ndarray,
                  scheme: ErrorCorrectionScheme = ErrorCorrectionScheme.IDEAL,
                  ec_efficiency: float = 1.16) -> ECResult:
    """
    Simulate the error-correction step.

    In a real implementation this would run the Cascade or LDPC algorithm.
    Here we model the *information leakage* rather than the bit operations:

    * **Ideal**: Bob's bits are replaced by Alice's; bits revealed = f_EC × H₂(e) × n.
    * **Cascade / LDPC**: same leakage model but with the appropriate
      efficiency factor.
    * **None**: no correction; Bob keeps his erroneous bits.

    The number of bits revealed to Eve (``parity_bits_revealed``) reduces
    the privacy amplification output.

    Parameters
    ----------
    alice_bits, bob_bits : ndarray of int
    scheme : ErrorCorrectionScheme
    ec_efficiency : float
        f_EC ≥ 1.0 (Cascade ≈ 1.16, LDPC near-ideal ≈ 1.05).

    Returns
    -------
    ECResult
    """
    n     = len(alice_bits)
    if n == 0:
        return ECResult(corrected_bob=np.array([], dtype=int), parity_bits_revealed=0)

    if scheme == ErrorCorrectionScheme.NONE:
        return ECResult(corrected_bob=bob_bits.copy(),
                        parity_bits_revealed=0)

    # Estimate bit-error rate
    errors = int(np.sum(alice_bits != bob_bits))
    e      = errors / n

    # Leakage: f_EC × H₂(e) × n bits
    h2     = _binary_entropy(e)
    leakage = int(math.ceil(ec_efficiency * h2 * n))

    if scheme == ErrorCorrectionScheme.IDEAL:
        corrected = alice_bits.copy()   # perfect correction
    else:
        # Cascade / LDPC: still assume perfect correction for key bits
        corrected = alice_bits.copy()

    return ECResult(
        corrected_bob          = corrected,
        parity_bits_revealed   = leakage,
        success                = True,
    )


# ---------------------------------------------------------------------------
# 4. Privacy amplification
# ---------------------------------------------------------------------------

def privacy_amplify(bits: np.ndarray,
                    qber: float,
                    parity_bits_revealed: int,
                    ec_efficiency: float,
                    security_epsilon: float) -> PAResult:
    """
    Compress the reconciled key to extract a shorter, provably secure key.

    Secret key length (Devetak-Winter / Shor-Preskill bound)::

        l = n · [1 − h(e)] − f_EC · n · h(e) − 2·log₂(1/ε) − correction

    simplified to::

        l = n · [1 − (1 + f_EC) · h(e)] − 2·log₂(1/ε)

    The actual compression uses a universal hash (SHA-256 derived) of the
    appropriate output length.

    Parameters
    ----------
    bits : ndarray of int
        Reconciled key bits.
    qber : float
        Estimated QBER.
    parity_bits_revealed : int
        Bits leaked during error correction.
    ec_efficiency : float
    security_epsilon : float

    Returns
    -------
    PAResult
    """
    n   = len(bits)
    result = PAResult()

    if n == 0:
        return result

    h   = _binary_entropy(qber)
    # Available secret key bits
    security_bits = 2.0 * math.log2(1.0 / max(security_epsilon, 1e-300))
    l = int(n * (1.0 - h) - parity_bits_revealed - security_bits)
    l = max(0, l)

    if l == 0:
        return result

    # Compress with a cryptographic hash (Toeplitz-like via SHA-256)
    raw_bytes    = np.packbits(bits).tobytes()
    hash_bytes   = hashlib.sha256(raw_bytes).digest()   # 256 bits
    # Extend/truncate using multiple hash rounds
    full_hash_bits = _extend_hash(raw_bytes, l)

    result.secret_key        = full_hash_bits[:l]
    result.secret_key_bits   = l
    result.pa_compression    = l / n
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _binary_entropy(e: float) -> float:
    """H₂(e) = −e·log₂(e) − (1−e)·log₂(1−e)."""
    if e <= 0.0 or e >= 1.0:
        return 0.0
    return -e * math.log2(e) - (1 - e) * math.log2(1 - e)


def _extend_hash(data: bytes, n_bits: int) -> np.ndarray:
    """Produce n_bits of output using iterated SHA-256."""
    bits_out = []
    counter  = 0
    while len(bits_out) < n_bits:
        h = hashlib.sha256(data + counter.to_bytes(4, "big")).digest()
        bits_out.extend(np.unpackbits(np.frombuffer(h, dtype=np.uint8)).tolist())
        counter += 1
    arr = np.array(bits_out[:n_bits], dtype=np.uint8)
    return arr


def compute_secret_key_rate(n_raw: int,
                             qber: float,
                             ec_efficiency: float = 1.16,
                             security_epsilon: float = 1e-10) -> float:
    """
    Compute the theoretical secret key rate (bits per raw photon) using the
    asymptotic Shor-Preskill formula::

        r = 1 − (1 + f_EC) · H₂(QBER)

    Clamped to [0, 1].
    """
    if qber >= 0.11:
        return 0.0
    h = _binary_entropy(qber)
    return max(0.0, 1.0 - (1.0 + ec_efficiency) * h)


# ---------------------------------------------------------------------------
# Full pipeline function
# ---------------------------------------------------------------------------

def run_postprocessing(
        alice_bits:   np.ndarray,
        alice_bases:  np.ndarray,
        bob_bits:     np.ndarray,
        bob_bases:    np.ndarray,
        detected_mask: np.ndarray,
        config: ProtocolConfig,
        rng: np.random.Generator,
) -> PostProcessingResult:
    """
    Execute the complete post-processing pipeline.

    Parameters
    ----------
    alice_bits, alice_bases : shape (N,) int arrays
    bob_bits, bob_bases     : shape (N,) int arrays
        ``bob_bits`` / ``bob_bases`` only meaningful for detected slots;
        undetected entries can be 0.
    detected_mask : shape (N,) bool
        True where Bob had a detection event.
    config : ProtocolConfig
    rng    : np.random.Generator

    Returns
    -------
    PostProcessingResult
    """
    out = PostProcessingResult()

    # ---- 1. Sifting -------------------------------------------------------
    sift_res       = sift(alice_bits, alice_bases, bob_bits, bob_bases, detected_mask)
    out.sifting    = sift_res

    if len(sift_res.sifted_alice) == 0:
        out.aborted      = True
        out.abort_reason = "empty sifted key"
        return out

    # ---- 2. QBER estimation -----------------------------------------------
    qber_res       = estimate_qber(
        sift_res.sifted_alice,
        sift_res.sifted_bob,
        sift_res.sifted_indices,
        config.sharing_rate,
        config.qber_threshold,
        rng,
    )
    out.qber = qber_res

    if qber_res.aborted:
        out.aborted      = True
        out.abort_reason = f"QBER={qber_res.estimated_qber:.4f} > threshold={config.qber_threshold}"
        return out

    remaining_alice = qber_res.remaining_alice
    remaining_bob   = qber_res.remaining_bob

    if len(remaining_alice) == 0:
        out.aborted      = True
        out.abort_reason = "no bits remaining after QBER sharing"
        return out

    # ---- 3. Error correction ----------------------------------------------
    if config.enable_error_correction:
        ec_res = error_correct(remaining_alice, remaining_bob,
                               config.ec_scheme, config.ec_efficiency)
    else:
        ec_res = ECResult(corrected_bob=remaining_bob.copy(),
                          parity_bits_revealed=0)
    out.ec = ec_res

    reconciled = ec_res.corrected_bob  # Alice's bits = ground truth after EC

    # ---- 4. Privacy amplification ----------------------------------------
    if config.enable_privacy_amplification:
        pa_res = privacy_amplify(
            reconciled,
            qber_res.estimated_qber,
            ec_res.parity_bits_revealed,
            config.ec_efficiency,
            config.security_epsilon,
        )
    else:
        pa_res = PAResult(
            secret_key      = reconciled.copy(),
            secret_key_bits = len(reconciled),
            pa_compression  = 1.0,
        )
    out.pa = pa_res

    out.secret_key      = pa_res.secret_key
    n_raw               = len(alice_bits)
    out.secret_key_rate = pa_res.secret_key_bits / n_raw if n_raw > 0 else 0.0

    return out
