#!/usr/bin/env python3
"""
nuqkd/examples/run_bb84.py
===========================
Canonical execution script for the NuQKD BB84 simulation.

Usage
-----
Run with default MEDIUM profile::

    python run_bb84.py

Run with a specific profile and seed::

    python run_bb84.py --profile hard --seed 42 --iterations 20

Run with a random profile (unpredictable — for AI red team testing)::

    python run_bb84.py --profile random

Run with PNS attack active::

    python run_bb84.py --profile medium --attack pns

Run with custom hardware config (SNSPD instead of SPAD)::

    python run_bb84.py --detector snspd --profile expert

Profiles: clean | easy | medium | hard | expert | random
Attacks:  none  | ir   | ir_optimal | pns
Hardware: spad  | snspd
"""

import argparse
import logging
import sys
from pathlib import Path

# ── Path setup (allows running from any directory) ────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from nuqkd.config.parameters import (
    DetectorConfig,
    ProtocolConfig,
    SimulationConfig,
    SourceConfig,
    SourceType,
)
from nuqkd.config.vulnerability_profiles import NAMED_PROFILES, get_random_profile
from nuqkd.core.session import QKDSession
from nuqkd.attacks.intercept_resend import InterceptResendAttack, OptimalIRAttack
from nuqkd.attacks.pns import PNSAttack

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nuqkd.run")


# ---------------------------------------------------------------------------
# Hardware presets
# ---------------------------------------------------------------------------

def build_detector(name: str) -> DetectorConfig:
    if name == "snspd":
        return DetectorConfig(
            efficiency         = 0.92,
            dark_count_rate_hz = 5.0,
            dead_time_ns       = 50.0,
            timing_jitter_ps   = 25.0,
            afterpulse_prob    = 0.0,
            insertion_loss_db  = 1.0,
        )
    elif name == "spad":
        return DetectorConfig(
            efficiency         = 0.25,
            dark_count_rate_hz = 1000.0,
            dead_time_ns       = 20_000.0,
            timing_jitter_ps   = 400.0,
            afterpulse_prob    = 0.005,
            insertion_loss_db  = 3.0,
        )
    elif name == "default":
        return DetectorConfig()   # library defaults
    else:
        raise ValueError(f"Unknown detector preset: {name!r}")


# ---------------------------------------------------------------------------
# Attack factory
# ---------------------------------------------------------------------------

def build_attack(name: str, T: float):
    """Build an attack agent given a name and channel transmittance."""
    if name == "none" or name is None:
        return None
    elif name == "ir":
        logger.info("Attack: Intercept-and-Resend (ε=1.0)")
        return InterceptResendAttack(intercept_rate=1.0, random_basis=True)
    elif name == "ir_partial":
        logger.info("Attack: Intercept-and-Resend (ε=0.3)")
        return InterceptResendAttack(intercept_rate=0.30, random_basis=True)
    elif name == "ir_optimal":
        logger.info("Attack: Optimal IR (Breidbart basis)")
        return OptimalIRAttack(intercept_rate=1.0)
    elif name == "pns":
        logger.info("Attack: PNS (T=%.4f)", T)
        return PNSAttack(channel_transmittance=T, store_fraction=1.0)
    else:
        raise ValueError(f"Unknown attack: {name!r}")


# ---------------------------------------------------------------------------
# Validation test: PROFILE_CLEAN vs PROFILE_MEDIUM
# ---------------------------------------------------------------------------

def run_validation(seed: int = 42, n_iterations: int = 5) -> bool:
    """
    Verify that the ChannelObserver can distinguish a clean channel
    from a vulnerable one.  Returns True if the check passes.
    """
    logger.info("=" * 60)
    logger.info("  VALIDATION: Clean vs Medium profile")
    logger.info("=" * 60)

    results = {}
    for name in ["clean", "medium"]:
        cfg = SimulationConfig(verbose=False)
        cfg.protocol = ProtocolConfig(raw_key_size=5000, num_iterations=n_iterations)
        session = QKDSession.from_profile(name, config=cfg, seed=seed)
        report  = session.run()
        results[name] = report
        logger.info("\n%s", report.summary())

    # The key diagnostic: timing bimodality score
    bim_clean  = results["clean"].observer_data.get("timing_bimodality_score", 0)
    bim_medium = results["medium"].observer_data.get("timing_bimodality_score", 0)
    mu_disc    = results["medium"].observer_data.get("mu_discrepancy", 0)

    logger.info("\n── Validation Results ──────────────────────────────────")
    logger.info("  Bimodality (clean)  : %.4f", bim_clean)
    logger.info("  Bimodality (medium) : %.4f", bim_medium)
    logger.info("  μ discrepancy (med) : %.4f", mu_disc)

    passed = (bim_medium > bim_clean + 0.2) or (abs(mu_disc) > 0.05)
    status = "PASSED ✓" if passed else "FAILED ✗"
    logger.info("  Observer coherence  : %s", status)
    logger.info("─" * 60)
    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="NuQKD BB84 Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--profile",    default="medium",
                        choices=list(NAMED_PROFILES) + ["random"],
                        help="Vulnerability profile")
    parser.add_argument("--detector",   default="default",
                        choices=["default", "spad", "snspd"],
                        help="Bob's detector hardware preset")
    parser.add_argument("--attack",     default="none",
                        choices=["none", "ir", "ir_partial", "ir_optimal", "pns"],
                        help="Active eavesdropping attack")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Number of key-distribution rounds")
    parser.add_argument("--raw-key",    type=int, default=10_000,
                        help="Raw key size (photons per iteration)")
    parser.add_argument("--seed",       type=int, default=None,
                        help="RNG seed for reproducibility")
    parser.add_argument("--validate",   action="store_true",
                        help="Run clean-vs-medium validation test first")
    parser.add_argument("--verbose",    action="store_true",
                        help="Per-iteration logging")
    parser.add_argument("--output-dir", default="./results",
                        help="Directory for JSON reports")
    parser.add_argument("--mu",         type=float, default=0.5,
                        help="Mean photon number μ (declared)")
    parser.add_argument("--distance",   type=float, default=10.0,
                        help="Fibre distance [km]")
    args = parser.parse_args()

    # ── Optional validation run ───────────────────────────────────────────
    if args.validate:
        ok = run_validation(seed=args.seed or 42)
        if not ok:
            logger.warning("Validation failed — observer may not detect vulnerabilities")

    # ── Build config ──────────────────────────────────────────────────────
    cfg = SimulationConfig(
        protocol_name  = "BB84",
        verbose        = args.verbose,
        export_results = True,
        output_dir     = args.output_dir,
    )
    cfg.source = SourceConfig(
        mean_photon_number = args.mu,
        enable_decoy       = True,
        type               = SourceType.WEAK_COHERENT_PULSE,
    )
    cfg.channel.fiber.distance_km = args.distance
    cfg.detector  = build_detector(args.detector)
    cfg.protocol  = ProtocolConfig(
        raw_key_size   = args.raw_key,
        num_iterations = args.iterations or 10,
        qber_threshold = 0.11,
    )

    # ── Build vulnerability profile ───────────────────────────────────────
    if args.profile == "random":
        vuln = get_random_profile(args.seed)
        profile_name = "random"
    else:
        vuln         = NAMED_PROFILES[args.profile]
        profile_name = args.profile

    # ── Estimate T for attack calibration ────────────────────────────────
    T_est = 10 ** (-(
        cfg.channel.fiber.attenuation_db_per_km * cfg.channel.fiber.distance_km
        + cfg.channel.insertion_loss_db
        + cfg.detector.insertion_loss_db
    ) / 10.0)

    attack = build_attack(args.attack, T_est)

    # ── Run session ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  NuQKD BB84 Simulation")
    logger.info("  Profile:    %s", profile_name)
    logger.info("  Detector:   %s  (η=%.2f)", args.detector, cfg.detector.efficiency)
    logger.info("  Attack:     %s", args.attack)
    logger.info("  Distance:   %.1f km  (T_est=%.4f)", args.distance, T_est)
    logger.info("  μ:          %.2f", args.mu)
    logger.info("  N:          %d pulses × %d iters",
                cfg.protocol.raw_key_size, cfg.protocol.num_iterations)
    logger.info("  Seed:       %s", args.seed)
    logger.info("=" * 60)

    session = QKDSession(
        config       = cfg,
        vuln_profile = vuln,
        profile_name = profile_name,
        attack_agent = attack,
        seed         = args.seed,
    )

    report = session.run()
    print(report.summary())

    # Print attack statistics if applicable
    if attack is not None:
        stats = attack.statistics()
        logger.info("\n── Attack Statistics ────────────────────────────────")
        for k, v in stats.items():
            logger.info("  %-35s: %s", k, v)

    return report


if __name__ == "__main__":
    main()
