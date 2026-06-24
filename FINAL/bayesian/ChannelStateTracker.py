import json
import os
import math
from datetime import datetime
from typing import Dict, Any, List, Optional


# ============================================================================
# Conjugate Prior Helpers
# ============================================================================

class BetaDistribution:
    """
    Beta-Bernoulli conjugate prior per parametri binomiali (QBER).

    Prior:  Beta(α, β)
    Likelihood: Bernoulli(p) → k errori su n trial
    Posterior: Beta(α + k, β + n - k)
    """

    def __init__(self, alpha: float = 2.0, beta_param: float = 18.0):
        self.alpha = alpha
        self.beta = beta_param

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        s = self.alpha + self.beta
        return (self.alpha * self.beta) / (s ** 2 * (s + 1))

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def update(self, k: int, n: int) -> 'BetaDistribution':
        """Aggiorna con k errori su n trial. Returns una nuova BetaDistribution."""
        return BetaDistribution(alpha=self.alpha + k, beta_param=self.beta + n - k)

    def to_dict(self) -> Dict[str, float]:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "mean": self.mean,
            "std": self.std,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> 'BetaDistribution':
        return cls(alpha=d["alpha"], beta_param=d["beta"])


class NormalNormalConjugate:
    """
    Normal-Normal conjugate prior per parametri continui (purity, damping params).

    Prior:  μ ~ Normal(μ₀, τ₀²)
    Likelihood: x ~ Normal(μ, σ²)   (σ² noto)
    Posterior: μ | x ~ Normal(μₙ, τₙ²)
    """

    def __init__(self, mu0: float = 0.5, tau0_sq: float = 1.0, known_sigma_sq: float = 0.04):
        self.mu0 = mu0
        self.tau0_sq = tau0_sq
        self.known_sigma_sq = known_sigma_sq
        self.n_observations = 0
        self.sum_observations = 0.0

    @property
    def posterior_precision(self) -> float:
        return 1.0 / self.tau0_sq + self.n_observations / self.known_sigma_sq

    @property
    def posterior_variance(self) -> float:
        return 1.0 / self.posterior_precision if self.posterior_precision > 0 else self.tau0_sq

    @property
    def posterior_mean(self) -> float:
        if self.n_observations == 0:
            return self.mu0
        prior_weight = self.mu0 / self.tau0_sq
        data_weight = (self.sum_observations / self.n_observations) * (self.n_observations / self.known_sigma_sq)
        return (prior_weight + data_weight) / self.posterior_precision

    @property
    def posterior_std(self) -> float:
        return math.sqrt(self.posterior_variance)

    def update(self, observations: List[float]) -> 'NormalNormalConjugate':
        """Aggiorna con nuove osservazioni."""
        nn = NormalNormalConjugate(
            mu0=self.mu0, tau0_sq=self.tau0_sq, known_sigma_sq=self.known_sigma_sq
        )
        nn.n_observations = self.n_observations + len(observations)
        nn.sum_observations = self.sum_observations + sum(observations)
        return nn

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mu0": self.mu0,
            "tau0_sq": self.tau0_sq,
            "known_sigma_sq": self.known_sigma_sq,
            "n_observations": self.n_observations,
            "sum_observations": self.sum_observations,
            "posterior_mean": self.posterior_mean,
            "posterior_std": self.posterior_std,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'NormalNormalConjugate':
        nn = cls(mu0=d["mu0"], tau0_sq=d["tau0_sq"], known_sigma_sq=d["known_sigma_sq"])
        nn.n_observations = d["n_observations"]
        nn.sum_observations = d["sum_observations"]
        return nn


# ============================================================================
# Channel State Tracker
# ============================================================================

DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "channel", "data", "channel_state.json"
)


class ChannelStateTracker:
    """
    Tracker bayesiano dello stato del canale quantistico BB84.

    Mantiene distribuzioni prior → posterior per i parametri del canale,
    un anomaly baseline adattivo (finestra mobile ultimi 5 episodi),
    e un registry di efficacia delle strategie.

    Persiste lo stato su JSON tra le sessioni.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._ensure_db_dir()

        # --- Parameter distributions (conjugate priors) ---
        self.qber_dist = BetaDistribution(alpha=2.0, beta_param=18.0)
        self.purity_dist = NormalNormalConjugate(mu0=0.95, tau0_sq=0.04, known_sigma_sq=0.01)
        self.depol_dist = NormalNormalConjugate(mu0=0.02, tau0_sq=0.01, known_sigma_sq=0.005)
        self.ampl_damp_dist = NormalNormalConjugate(mu0=0.10, tau0_sq=0.04, known_sigma_sq=0.01)
        self.phase_damp_dist = NormalNormalConjugate(mu0=0.03, tau0_sq=0.02, known_sigma_sq=0.005)

        # --- Anomaly baseline (sliding window: last 5 episodes) ---
        self.qber_history: List[float] = []
        self.purity_history: List[float] = []
        self.sifted_len_history: List[int] = []
        self.WINDOW_SIZE = 5

        # --- Strategy effectiveness registry ---
        self.strategy_registry: Dict[str, List[Dict[str, Any]]] = {}

        # --- Episode counter ---
        self.total_episodes = 0

        # Load persisted state if available
        self._load_state()

    def _ensure_db_dir(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_state(self):
        """Salva lo stato corrente su JSON."""
        state = {
            "metadata": {
                "saved_at": datetime.utcnow().isoformat(),
                "total_episodes": self.total_episodes,
                "version": "1.0",
            },
            "qber_dist": self.qber_dist.to_dict(),
            "purity_dist": self.purity_dist.to_dict(),
            "depol_dist": self.depol_dist.to_dict(),
            "ampl_damp_dist": self.ampl_damp_dist.to_dict(),
            "phase_damp_dist": self.phase_damp_dist.to_dict(),
            "qber_history": self.qber_history[-self.WINDOW_SIZE:],
            "purity_history": self.purity_history[-self.WINDOW_SIZE:],
            "sifted_len_history": self.sifted_len_history[-self.WINDOW_SIZE:],
            "strategy_registry": self.strategy_registry,
        }

        tmp_path = self.db_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, self.db_path)

    def _load_state(self):
        """Carica lo stato persistente se disponibile."""
        if not os.path.exists(self.db_path):
            return

        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                state = json.load(f)

            self.qber_dist = BetaDistribution.from_dict(state["qber_dist"])
            self.purity_dist = NormalNormalConjugate.from_dict(state["purity_dist"])
            self.depol_dist = NormalNormalConjugate.from_dict(state["depol_dist"])
            self.ampl_damp_dist = NormalNormalConjugate.from_dict(state["ampl_damp_dist"])
            self.phase_damp_dist = NormalNormalConjugate.from_dict(state["phase_damp_dist"])
            self.qber_history = state.get("qber_history", [])
            self.purity_history = state.get("purity_history", [])
            self.sifted_len_history = state.get("sifted_len_history", [])
            self.strategy_registry = state.get("strategy_registry", {})
            self.total_episodes = state["metadata"].get("total_episodes", 0)

        except Exception:
            pass

    # ------------------------------------------------------------------
    # Main Update (Bayesian update dopo ogni episodio)
    # ------------------------------------------------------------------

    def update(self, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggiorna lo stato del canale con i dati di un nuovo episodio.

        Parameters
        ----------
        episode_data : dict
            Deve contenere:
              - episode: int
              - baseline_qber: float (QBER senza attacco)
              - attack_qber: float (QBER durante attacco)
              - baseline_purity: float (purity media senza attacco)
              - attack_purity: float (purity media durante attacco)
              - sifted_len_baseline: int
              - detected: bool
              - executed_strategy: str (nome strategia usata)
              - interception_rate: float
              - pns_enabled: bool
              - parameter_estimation: dict (dal recon agent, opzionale)

        Returns
        -------
        Dict[str, Any]
            Stato aggiornato con posterior summaries.
        """
        self.total_episodes += 1

        # --- Update QBER distribution (Beta-Bernoulli) ---
        baseline_qber = episode_data.get("baseline_qber", 0.05)
        sifted_len = episode_data.get("sifted_len_baseline", 1000)
        errors_baseline = int(baseline_qber * sifted_len) if sifted_len > 0 else 0

        self.qber_dist = self.qber_dist.update(k=errors_baseline, n=sifted_len)

        # --- Update Purity distribution (Normal-Normal) ---
        baseline_purity = episode_data.get("baseline_purity")
        attack_purity = episode_data.get("attack_purity")
        purity_obs = []
        if baseline_purity is not None:
            purity_obs.append(baseline_purity)
        if attack_purity is not None and attack_purity == attack_purity:
            purity_obs.append(attack_purity)

        if purity_obs:
            self.purity_dist = self.purity_dist.update(purity_obs)

        # --- Update Depolarization / Damping distributions ---
        pe = episode_data.get("parameter_estimation", {})
        if pe:
            depol_val = pe.get("depolarization_probability")
            if depol_val is not None:
                self.depol_dist = self.depol_dist.update([depol_val])

            ampl_val = pe.get("amplitude_damping_gamma")
            if ampl_val is not None:
                self.ampl_damp_dist = self.ampl_damp_dist.update([ampl_val])

            phase_val = pe.get("phase_damping_lambda")
            if phase_val is not None:
                self.phase_damp_dist = self.phase_damp_dist.update([phase_val])

        # --- Update anomaly baseline (sliding window) ---
        self.qber_history.append(baseline_qber)
        if len(self.qber_history) > self.WINDOW_SIZE:
            self.qber_history = self.qber_history[-self.WINDOW_SIZE:]

        if baseline_purity is not None:
            self.purity_history.append(baseline_purity)
            if len(self.purity_history) > self.WINDOW_SIZE:
                self.purity_history = self.purity_history[-self.WINDOW_SIZE:]

        if sifted_len > 0:
            self.sifted_len_history.append(sifted_len)
            if len(self.sifted_len_history) > self.WINDOW_SIZE:
                self.sifted_len_history = self.sifted_len_history[-self.WINDOW_SIZE:]

        # --- Update strategy effectiveness registry ---
        strategy_name = episode_data.get("executed_strategy", "unknown")
        detected = episode_data.get("detected", False)
        success = not detected

        entry = {
            "episode": episode_data.get("episode", self.total_episodes),
            "success": success,
            "interception_rate": episode_data.get("interception_rate", 0.0),
            "pns_enabled": episode_data.get("pns_enabled", False),
            "baseline_qber": baseline_qber,
            "attack_qber": episode_data.get("attack_qber"),
            "qber_delta": (episode_data.get("attack_qber", 0) - baseline_qber)
                          if episode_data.get("attack_qber") is not None else 0.0,
        }

        if strategy_name not in self.strategy_registry:
            self.strategy_registry[strategy_name] = []
        self.strategy_registry[strategy_name].append(entry)

        # --- Persist state ---
        self._save_state()

        return self.get_channel_summary()

    # ------------------------------------------------------------------
    # Priors Generator (per il prossimo Recon Agent)
    # ------------------------------------------------------------------

    def get_priors(self) -> Dict[str, Any]:
        """
        Restituisce priors informati per il recon agent dell'episodio successivo.
        I priors sono basati sulle posterior correnti del tracker.
        """
        return {
            "qber_prior": {"mean": self.qber_dist.mean, "std": self.qber_dist.std},
            "purity_prior": {"mean": self.purity_dist.posterior_mean, "std": self.purity_dist.posterior_std},
            "depol_prior": {"mean": self.depol_dist.posterior_mean, "std": self.depol_dist.posterior_std},
            "ampl_damp_prior": {"mean": self.ampl_damp_dist.posterior_mean, "std": self.ampl_damp_dist.posterior_std},
            "phase_damp_prior": {"mean": self.phase_damp_dist.posterior_mean, "std": self.phase_damp_dist.posterior_std},
        }

    # ------------------------------------------------------------------
    # Anomaly Baseline (dinamico)
    # ------------------------------------------------------------------

    def get_anomaly_baseline(self) -> Dict[str, Any]:
        """
        Restituisce il baseline dinamico per l'anomaly detection.
        Calcolato come media e deviazione standard degli ultimi 5 episodi.
        """
        qber_mean = sum(self.qber_history) / len(self.qber_history) if self.qber_history else 0.05
        qber_std = (sum((x - qber_mean) ** 2 for x in self.qber_history) / max(len(self.qber_history), 1)) ** 0.5

        purity_mean = sum(self.purity_history) / len(self.purity_history) if self.purity_history else 0.95
        purity_std = (sum((x - purity_mean) ** 2 for x in self.purity_history) / max(len(self.purity_history), 1)) ** 0.5

        return {
            "qber_baseline": {
                "mean": qber_mean,
                "std": qber_std if len(self.qber_history) > 1 else 0.02,
                "min": min(self.qber_history) if self.qber_history else 0.03,
                "max": max(self.qber_history) if self.qber_history else 0.07,
            },
            "purity_baseline": {
                "mean": purity_mean,
                "std": purity_std if len(self.purity_history) > 1 else 0.05,
                "min": min(self.purity_history) if self.purity_history else 0.85,
                "max": max(self.purity_history) if self.purity_history else 0.99,
            },
            "sifted_len_baseline": {
                "mean": sum(self.sifted_len_history) / len(self.sifted_len_history) if self.sifted_len_history else 1500,
            },
            "window_size": min(len(self.qber_history), self.WINDOW_SIZE),
        }

    def compute_anomaly_score(self, value: float, baseline_mean: float, baseline_std: float) -> float:
        """Calcola un anomaly score normalizzato (0.0-1.0)."""
        if baseline_std < 1e-9:
            baseline_std = 0.02
        z_score = abs(value - baseline_mean) / baseline_std
        return min(z_score / 3.0, 1.0)

    # ------------------------------------------------------------------
    # Strategy Effectiveness Registry
    # ------------------------------------------------------------------

    def get_strategy_effectiveness(self) -> Dict[str, Any]:
        """Restituisce statistiche aggregate sull'efficacia delle strategie usate."""
        result = {}
        for strategy_name, entries in self.strategy_registry.items():
            successes = sum(1 for e in entries if e["success"])
            qber_deltas = [e["qber_delta"] for e in entries if e["qber_delta"] == e["qber_delta"]]

            result[strategy_name] = {
                "total_attempts": len(entries),
                "successful_attacks": successes,
                "success_rate": successes / len(entries) if entries else 0.0,
                "avg_qber_delta": sum(qber_deltas) / len(qber_deltas) if qber_deltas else 0.0,
                "avg_interception_rate": sum(e["interception_rate"] for e in entries) / len(entries),
            }

        return result

    def get_best_strategy_for_conditions(
        self, current_qber: float, belief: float
    ) -> Optional[Dict[str, Any]]:
        """Suggerisce la strategia più efficace per le condizioni correnti del canale."""
        best_strategy = None
        best_score = -1.0

        for strategy_name, entries in self.strategy_registry.items():
            similar_entries = [
                e for e in entries
                if abs(e["baseline_qber"] - current_qber) < 0.03
            ]

            if not similar_entries:
                continue

            success_rate = sum(1 for e in similar_entries if e["success"]) / len(similar_entries)
            avg_delta = sum(abs(e["qber_delta"]) for e in similar_entries) / len(similar_entries)

            score = 0.6 * success_rate + 0.3 * (1 - min(avg_delta, 1.0)) + 0.1 * belief

            if score > best_score:
                best_score = score
                avg_ir = sum(e["interception_rate"] for e in similar_entries) / len(similar_entries)
                pns_count = sum(1 for e in similar_entries if e["pns_enabled"])
                best_strategy = {
                    "strategy_name": strategy_name,
                    "success_rate_in_similar_conditions": success_rate,
                    "recommended_interception_rate": avg_ir,
                    "recommended_pns_enabled": pns_count > len(similar_entries) / 2,
                    "confidence": score,
                    "similar_episodes_count": len(similar_entries),
                }

        return best_strategy

    # ------------------------------------------------------------------
    # Channel Summary (stato completo)
    # ------------------------------------------------------------------

    def get_channel_summary(self) -> Dict[str, Any]:
        """Restituisce uno snapshot completo dello stato del canale."""
        anomaly_baseline = self.get_anomaly_baseline()
        priors = self.get_priors()

        return {
            "total_episodes": self.total_episodes,
            "parameter_posteriors": {
                "qber": {"mean": self.qber_dist.mean, "std": self.qber_dist.std},
                "purity": {"mean": self.purity_dist.posterior_mean, "std": self.purity_dist.posterior_std},
                "depolarization": {"mean": self.depol_dist.posterior_mean, "std": self.depol_dist.posterior_std},
                "amplitude_damping": {"mean": self.ampl_damp_dist.posterior_mean, "std": self.ampl_damp_dist.posterior_std},
                "phase_damping": {"mean": self.phase_damp_dist.posterior_mean, "std": self.phase_damp_dist.posterior_std},
            },
            "anomaly_baseline": anomaly_baseline,
            "priors_for_next_recon": priors,
            "strategy_effectiveness": self.get_strategy_effectiveness(),
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self):
        """Reset completo dello stato (priori iniziali)."""
        self.qber_dist = BetaDistribution(alpha=2.0, beta_param=18.0)
        self.purity_dist = NormalNormalConjugate(mu0=0.95, tau0_sq=0.04, known_sigma_sq=0.01)
        self.depol_dist = NormalNormalConjugate(mu0=0.02, tau0_sq=0.01, known_sigma_sq=0.005)
        self.ampl_damp_dist = NormalNormalConjugate(mu0=0.10, tau0_sq=0.04, known_sigma_sq=0.01)
        self.phase_damp_dist = NormalNormalConjugate(mu0=0.03, tau0_sq=0.02, known_sigma_sq=0.005)
        self.qber_history.clear()
        self.purity_history.clear()
        self.sifted_len_history.clear()
        self.strategy_registry.clear()
        self.total_episodes = 0
        self._save_state()
