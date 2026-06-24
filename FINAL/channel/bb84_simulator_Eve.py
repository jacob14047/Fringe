import numpy as np
import secrets
import math
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional, Union
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from enum import Enum
import time



# ============================================================================
# Utility functions
# ============================================================================

def random_bit() -> int:
    """Cryptographically secure random bit."""
    return secrets.randbits(1)


def random_bits(n: int) -> np.ndarray:
    """Array of n random bits (using numpy for efficiency)."""
    return np.random.randint(0, 2, size=n, dtype=np.uint8)


def random_bases(n: int) -> np.ndarray:
    """0 = rectilinear (Z), 1 = diagonal (X)."""
    return np.random.randint(0, 2, size=n, dtype=np.uint8)


# ============================================================================
# Enums and data structures
# ============================================================================

class Basis(Enum):
    Z = 0  # rectilinear (horizontal/vertical)
    X = 1  # diagonal (45°/135°)


# ============================================================================
# Security analysis module
# ============================================================================

class VisualizationSuite:
    """Grafici per canale BB84 pulito (senza Eve)."""

    def plot_qber_vs_noise(self, iterations: int = 30):
        """QBER al variare della depolarizzazione."""
        depols = np.linspace(0, 0.15, 20)
        qbers = []

        for p in depols:
            config = SimulationConfig(
                raw_key_size=4000,
                num_iterations=1,
                depolarization_prob=p,
                use_amplitude_damping=True,
                amplitude_damping_gamma=0.1,
                use_phase_damping=True,
                phase_damping_lambda=0.05,
                track_state_purities=False
            )
            sim = BB84SimulationV2(config)
            result = sim.run_single_iteration_v2(0)
            qbers.append(result['qber_est'] if result['qber_est'] == result['qber_est'] else 0)

        plt.figure(figsize=(10, 6))
        plt.plot(depols * 100, np.array(qbers) * 100, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Depolarization Probability (%)', fontsize=12)
        plt.ylabel('QBER (%)', fontsize=12)
        plt.title('QBER vs Depolarization (clean channel, no Eve)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('qber_vs_noise.pdf', dpi=300)
        plt.show()
        print("✅ Salvato: qber_vs_noise.pdf")

    def plot_purity_vs_gamma(self):
        """Purità media degli stati al variare di amplitude damping γ."""
        gammas = np.linspace(0, 0.5, 20)
        purities = []

        for g in gammas:
            config = SimulationConfig(
                raw_key_size=1000,
                num_iterations=1,
                use_amplitude_damping=True,
                amplitude_damping_gamma=g,
                track_state_purities=True
            )
            sim = BB84SimulationV2(config)
            result = sim.run_single_iteration_v2(0)
            purities.append(result['avg_purity'] if result['avg_purity'] else 1.0)

        plt.figure(figsize=(10, 6))
        plt.plot(gammas, purities, 's-', linewidth=2, markersize=8, color='purple')
        plt.xlabel('Amplitude Damping γ', fontsize=12)
        plt.ylabel('Average State Purity', fontsize=12)
        plt.title('State Purity Degradation vs Channel Loss', fontsize=14)
        plt.ylim(0.5, 1.05)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('purity_vs_gamma.pdf', dpi=300)
        plt.show()
        print("✅ Salvato: purity_vs_gamma.pdf")

    def plot_key_distribution(self, n_iterations: int = 100):
        """Distribuzione della lunghezza della sifted key su N iterazioni."""
        config = SimulationConfig(
            raw_key_size=4000,
            num_iterations=n_iterations,
            use_amplitude_damping=True,
            amplitude_damping_gamma=0.1,
            use_phase_damping=True,
            phase_damping_lambda=0.05,
            depolarization_prob=0.01,
            track_state_purities=False
        )
        sim = BB84SimulationV2(config)
        sim.run_single_iteration_v2(0)  # Run all iterations to collect key lengths

        key_lengths = sim.results['sifted_key_lengths']
        mean_len = np.mean(key_lengths)

        plt.figure(figsize=(10, 6))
        plt.hist(key_lengths, bins=20, edgecolor='black', alpha=0.7)
        plt.axvline(mean_len, color='r', linestyle='--', linewidth=2,
                    label=f'Mean: {mean_len:.0f} bits')
        plt.xlabel('Sifted Key Length (bits)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Sifted Key Length Distribution ({n_iterations} iterations)', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('key_distribution.pdf', dpi=300)
        plt.show()
        print(f"✅ Salvato: key_distribution.pdf  |  Mean: {mean_len:.0f} bits")

    def plot_qber_vs_distance(self):
        """QBER al variare della distanza (tramite amplitude damping realistico)."""
        distances_km = np.linspace(0, 50, 20)
        # 0.2 dB/km → trasmittanza = 10^(-0.2d/10), gamma = 1 - trasmittanza
        gammas = 1 - 10 ** (-0.2 * distances_km / 10)
        qbers = []

        for g in gammas:
            config = SimulationConfig(
                raw_key_size=2000,
                num_iterations=1,
                use_amplitude_damping=True,
                amplitude_damping_gamma=float(g),
                depolarization_prob=0.01,
                track_state_purities=False
            )
            sim = BB84SimulationV2(config)
            result = sim.run_single_iteration_v2(0)
            qbers.append(result['qber_est'] if result['qber_est'] == result['qber_est'] else 0)

        plt.figure(figsize=(10, 6))
        plt.plot(distances_km, np.array(qbers) * 100, 'o-', linewidth=2,
                 markersize=8, color='darkorange')
        plt.xlabel('Distance (km)', fontsize=12)
        plt.ylabel('QBER (%)', fontsize=12)
        plt.title('QBER vs Fiber Distance (0.2 dB/km, no Eve)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('qber_vs_distance.pdf', dpi=300)
        plt.show()
        print("✅ Salvato: qber_vs_distance.pdf")

class SecurityAnalyzer:
    """
    Analisi rigorosa della sicurezza usando risultati provati.
    """
    
    def __init__(self, n_raw: int, qber: float, sharing_rate: float):
        """
        n_raw: numero di qubit grezzi
        qber: Quantum Bit Error Rate stimato
        sharing_rate: frazione di chiave rivelata per test
        """
        self.N = n_raw
        self.e = qber  # ε in literature
        self.f = sharing_rate
    
    @staticmethod
    def shor_preskill_bound(e_qber: float, threshold: float = 0.11) -> float:
        """
        Bound di Shor-Preskill (2000):
        
        Se QBER > h(threshold) dove h(x) = -x log₂(x) - (1-x) log₂(1-x)
        allora Eve non può avere ottenuto più di una certa informazione.
        
        Returns: Upper bound su Eve's information (bits)
        """
        import math
        
        def binary_entropy(x):
            if x <= 0 or x >= 1:
                return 0
            return -x * math.log2(x) - (1 - x) * math.log2(1 - x)
        
        # Se e_qber è basso, Eve ha info limitata
        h_e = binary_entropy(e_qber)
        
        # Information Eve could have gained (theoretical bound)
        # I_eve ≤ n_sifted * h(QBER)
        return h_e  # Per bit
    
    def calculate_secure_key_length(self) -> Tuple[int, float]:
        """
        Calcola lunghezza della chiave finale dopo:
        1. Error correction
        2. Privacy amplification
        
        Basato su: Bennett, Brassard, Crépeau, Maurer (1995)
        
        l ≥ d - 2 log₂(1/ε_sec) - 2
        dove:
          d = number of sifted bits
          ε_sec = security parameter (e.g., 2^-128)
        """
        n_sifted = int(self.N * 0.5)  # 50% sifting efficiency
        n_tested = int(n_sifted * self.f)  # Bits tested for QBER
        n_private = n_sifted - n_tested  # Bits for final key
        
        # Privacy amplification overhead
        eps_sec = 1e-30  # 2^-128 security level
        privacy_amp_cost = 2 * np.log2(1 / eps_sec) + 2
        
        secure_length = max(0, n_private - privacy_amp_cost)
        
        # Information Eve could have
        eve_info = self.shor_preskill_bound(self.e)
        eve_bits = int(n_tested * eve_info)
        
        return int(secure_length), eve_bits
    
    def test_hypothesis(self) -> Dict[str, float]:
        """
        Test statistico: Qual è il tasso di falso positivo/negativo?
        
        Usa Chernoff bound per quantificare:
        - P(reject|no eve) = False positive rate
        - P(accept|eve present) = False negative rate
        """
        n_tested = int(self.N * 0.5 * self.f)
        
        # H0: No Eve (p = 0.05, just noise)
        p0 = 0.05
        
        # H1: Eve (p = 0.12, noise + Eve)
        p1 = 0.12
        
        threshold = 0.11
        threshold_count = threshold * n_tested
        
        # Chernoff bound for false negatives
        # P(QBER < threshold | Eve) ≤ exp(-D(p1||threshold))
        def kullback_leibler(p, q):
            return p * np.log2(p/q) + (1-p) * np.log2((1-p)/(1-q))
        
        kl_eve = kullback_leibler(p1, threshold)
        false_neg_bound = 2 ** (-n_tested * kl_eve)  # Prob Eve not detected
        
        # False positives
        kl_clean = kullback_leibler(p0, threshold)
        false_pos_bound = 2 ** (-n_tested * kl_clean)
        
        return {
            'false_positive_rate': float(false_pos_bound),
            'false_negative_rate': float(false_neg_bound),
            'security_gap': threshold - p0,  # Margin from noise
            'detection_power': 1 - false_neg_bound
        }


@dataclass
class BlochVector:
    """
    Rappresenta uno stato di qubit sulla Bloch sphere.
    
    Ogni stato puro di un qubit può essere scritto come:
    |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩
    
    Sulla Bloch sphere:
    x = sin(θ)cos(φ)
    y = sin(θ)sin(φ)
    z = cos(θ)
    
    Punti speciali:
    - (0,0,1):   |0⟩ (North pole)
    - (0,0,-1):  |1⟩ (South pole)
    - (1,0,0):   |+⟩ = (|0⟩+|1⟩)/√2
    - (-1,0,0):  |-⟩ = (|0⟩-|1⟩)/√2
    """
    x: float = 0.0
    y: float = 0.0
    z: float = 1.0  # Default: |0⟩
    
    def __post_init__(self):
        pass
        

    def to_array(self) -> np.ndarray:
        """Ritorna il vettore come array numpy."""
        return np.array([self.x, self.y, self.z])
    
    @classmethod
    def from_basis_and_bit(cls, basis: int, bit: int) -> 'BlochVector':
        """
        Crea uno stato di Bloch da una base e un bit.
        
        Parametri:
        -----------
        basis : int
            0 = Z basis (rectilinear: |0⟩, |1⟩)
            1 = X basis (diagonal: |+⟩, |-⟩)
        
        bit : int
            0 o 1
        
        Returns:
        --------
        BlochVector instance
        
        Example:
        --------
        >>> state = BlochVector.from_basis_and_bit(0, 1)  # |1⟩ in Z basis
        >>> print(state)
        BlochVector(x=0.0, y=0.0, z=-1.0)
        """
        if basis == 0:  # Z basis
            # |0⟩ -> (0, 0, 1),  |1⟩ -> (0, 0, -1)
            return cls(0, 0, 1 if bit == 0 else -1)
        else:  # X basis (basis == 1)
            # |+⟩ -> (1, 0, 0),  |-⟩ -> (-1, 0, 0)
            return cls(1 if bit == 0 else -1, 0, 0)

    @classmethod
    def z_basis_0(cls) -> 'BlochVector':
        return cls(0.0, 0.0, 1.0)

    @classmethod
    def z_basis_1(cls) -> 'BlochVector':
        return cls(0.0, 0.0, -1.0)

    @classmethod
    def x_basis_plus(cls) -> 'BlochVector':
        return cls(1.0, 0.0, 0.0)

    @classmethod
    def x_basis_minus(cls) -> 'BlochVector':
        return cls(-1.0, 0.0, 0.0)

    def measure(self, measurement_basis: int) -> Tuple[int, float]:
        """
        Misura lo stato quantico in una base specificata.
        
        Usa la regola di Born: P(outcome|basis) = |⟨outcome|ψ⟩|²
        
        Sulla Bloch sphere: P(outcome) = (1 + ⟨ψ|M_outcome|ψ⟩) / 2
                           dove M_outcome è il proiettore sull'outcome
        
        Parametri:
        ----------
        measurement_basis : int
            0 = Misura in Z basis
            1 = Misura in X basis
        
        Returns:
        --------
        measured_bit : int (0 o 1)
        probability : float (probabilità dell'outcome ottenuto)
        
        Example:
        --------
        >>> state = BlochVector.from_basis_and_bit(0, 1)  # |1⟩
        >>> bit, prob = state.measure(0)  # Misura in Z basis
        >>> print(f"Measured: {bit}, Probability: {prob:.2f}")
        Measured: 1, Probability: 1.00
        
        >>> bit, prob = state.measure(1)  # Misura in X basis (base sbagliata!)
        >>> print(f"Measured: {bit}, Probability: {prob:.2f}")
        Measured: 0 or 1, Probability: 0.50  # Random!
        """
        if measurement_basis == 0:  # Z basis
            # Proiettore su |0⟩: (0, 0, 1)
            # Proiettore su |1⟩: (0, 0, -1)
            measurement_vector = np.array([0, 0, 1])
        else:  # X basis (measurement_basis == 1)
            # Proiettore su |+⟩: (1, 0, 0)
            # Proiettore su |-⟩: (-1, 0, 0)
            measurement_vector = np.array([1, 0, 0])
        
        # Born rule: p(0) = (1 + ⟨measurement_vector · bloch⟩) / 2
        state_vector = self.to_array()
        overlap = np.dot(state_vector, measurement_vector)
        
        # Probabilità di misurare |0⟩ nella base specificata
        p_zero = 0.5 * (1.0 + overlap)
        
        # Campiona da Bernoulli(p_zero)
        measured_bit = 0 if np.random.random() < p_zero else 1
        
        # Probabilità dell'outcome ottenuto
        probability = p_zero if measured_bit == 0 else (1.0 - p_zero)
        
        # COLLASSO della funzione d'onda
        # Dopo la misura, lo stato collassa al eigenstate misurato
        if measured_bit == 0:
            # Collassa a |0⟩ nella base di misura
            self.x = measurement_vector[0]
            self.y = measurement_vector[1]
            self.z = measurement_vector[2]
        else:
            # Collassa a |1⟩ nella base di misura
            self.x = -measurement_vector[0]
            self.y = -measurement_vector[1]
            self.z = -measurement_vector[2]
        
        return measured_bit, probability
    
    def apply_depolarization(self, p: float) -> None:
        """
        Applica rumore di depolarizzazione.
        
        Canale di depolarizzazione: ρ' = (1-p)ρ + p·I/2
        
        Sulla Bloch sphere, questo equivale a rimescolare lo stato
        con il centro della sfera (stato misto I/2):
        bloch_vector' = (1-p) * bloch_vector
        
        Parametri:
        ----------
        p : float
            Probabilità di depolarizzazione (0 ≤ p ≤ 1)
        
        Example:
        --------
        >>> state = BlochVector.from_basis_and_bit(0, 0)  # |0⟩
        >>> state.apply_depolarization(0.1)  # 10% depolarization
        >>> print(state)  # Lo stato si avvicina al centro
        BlochVector(x=0.0, y=0.0, z=0.9)
        """
        scale_factor = 1.0 - p
        self.x *= scale_factor
        self.y *= scale_factor
        self.z *= scale_factor
    
    def apply_amplitude_damping(self, gamma: float) -> None:
        """
        Applica amplitude damping (perdita di energia).
        
        Questo è il modello più realistico per canali quantistici reali:
        - Fotoni persi nell'atmosfera o nella fibra ottica
        - Energia che si dissipa
        
        Kraus operators:
        K₀ = |0⟩⟨0| + √(1-γ)|1⟩⟨1|
        K₁ = √γ|0⟩⟨1|
        
        Sulla Bloch sphere (per stati puri):
        x' = (1-γ)x
        y' = (1-γ)y
        z' = 2γ(1/2 + z/2) - 1 = γ(1+z) - 1
        
        Parametri:
        ----------
        gamma : float
            Decay rate (0 ≤ γ ≤ 1)
            Per fibra ottica a 10 km e 0.2 dB/km: γ ≈ 0.15-0.20
        """
        # Applica gli effetti
        self.x *= np.sqrt(1 - gamma)
        self.y *= np.sqrt(1 - gamma)
        # z si muove verso -1 (stato |1⟩)
        self.z = (1 - gamma) * self.z + gamma  # si muove verso +1 (|0⟩, ground state)
    
    def apply_phase_damping(self, lambda_param: float) -> None:
        """
        Applica phase damping (dephasing).
        
        Questo modella la perdita di coerenza quantistica:
        - Interazione con l'ambiente
        - Fluttuazioni di fase
        - Decoherence
        
        Kraus operators:
        K₀ = |0⟩⟨0| + √(1-λ)|1⟩⟨1|
        K₁ = √λ|1⟩⟨1|
        
        Sulla Bloch sphere:
        x' = (1-λ)x
        y' = (1-λ)y
        z' = z  (no change in z)
        
        Parametri:
        ----------
        lambda_param : float
            Dephasing rate
        """
        self.x *= (1 - lambda_param)
        self.y *= (1 - lambda_param)
        # z non cambia
    
    def distance_from_ideal(self, ideal_basis: int, ideal_bit: int) -> float:
        """
        Calcola la distanza di traccia (trace distance) dallo stato ideale.
        
        Misura quanto lo stato attuale è "lontano" da uno stato puro ideale.
        
        D = 1/2 ||ρ - σ||₁
        
        Su Bloch sphere: D = 1/2 ||bloch_vector - ideal_vector||
        
        Returns:
        --------
        distance : float (0 ≤ distance ≤ 1)
            0 = stato identico all'ideale
            1 = stato completamente ortogonale
        """
        ideal_state = BlochVector.from_basis_and_bit(ideal_basis, ideal_bit)
        diff = self.to_array() - ideal_state.to_array()
        return 0.5 * np.linalg.norm(diff)
    
    def purity(self) -> float:
        """
        Calcola la purità dello stato: Tr(ρ²).
        
        Su Bloch sphere: P = (1 + ||bloch||²) / 2
        
        Returns:
        --------
        purity : float (0 ≤ P ≤ 1)
            1 = stato puro
            1/2 = stato completamente misto
        
        Example:
        --------
        >>> state = BlochVector.from_basis_and_bit(0, 0)  # |0⟩ puro
        >>> print(f"Purity: {state.purity():.2f}")  # ~1.0
        Purity: 1.00
        
        >>> state.apply_depolarization(0.5)
        >>> print(f"Purity: {state.purity():.2f}")  # ~0.75
        Purity: 0.75
        """
        r_squared = self.x**2 + self.y**2 + self.z**2
        return (1.0 + r_squared) / 2.0
    
    def plot_bloch_sphere(self, title: str = "Bloch Sphere") -> None:
        """
        Visualizza lo stato sulla Bloch sphere (se matplotlib disponibile).
        
        Example:
        --------
        >>> state = BlochVector.from_basis_and_bit(0, 1)  # |1⟩
        >>> state.plot_bloch_sphere(title="State |1⟩")
        """
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Sfera unitaria (sfondo)
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='cyan')
        
        # Vettore dello stato
        ax.quiver(0, 0, 0, self.x, self.y, self.z, color='red', arrow_length_ratio=0.1, linewidth=2)
        
        # Assi
        ax.quiver(0, 0, 0, 1.5, 0, 0, color='red', alpha=0.3, linewidth=1)
        ax.quiver(0, 0, 0, 0, 1.5, 0, color='green', alpha=0.3, linewidth=1)
        ax.quiver(0, 0, 0, 0, 0, 1.5, color='blue', alpha=0.3, linewidth=1)
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)
        ax.set_xlabel('X (|+⟩)')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z (|0⟩)')
        ax.set_title(title)
        
        plt.tight_layout()
        return fig


@dataclass
class SimulationConfig:
    """Main configuration for the BB84 simulation."""
    
    raw_key_size: int = 4000          # Number of photons sent (N)
    num_iterations: int  = 10         # Number of key distribution rounds
    depolarization_prob: float = 0.0  # p: total depolarization probability
    sharing_rate: float = 0.2         # f: fraction of sifted key shared for QBER estimation
    use_amplitude_damping: bool = True
    amplitude_damping_gamma: float = 0.15
    use_phase_damping: bool = False
    phase_damping_lambda: float = 0.05
    track_state_purities: bool = True
    use_thermal_noise: bool = False
    thermal_ratio: float = 0.02          # Dark counts, thermal photons
    fiber_loss_db_per_km: float = 0.2    # Standard ITU: 0.2 dB/km @ 1550nm
    distance_km: float = 10.0

    
    use_weak_laser: bool = False      # Weak coherent pulse source
    mean_photon_num: float = 0.1      # μ: average photons per pulse
    source_frequency_hz: float = 1e6  # f: pulse generation rate (Hz)
    use_channel_attenuation: bool = False
    attenuation_coeff: float = 0.2    # dB/km
    distance_km: float = 0.0
    detector_efficiency: float = 1.0  # η_D
    use_dead_time: bool = True
    dead_time_us: float = 0.01      # τ in microseconds
    
    interception_rate: float = 0.0      # ε: frazione di fotoni intercettati (0..1)
    eve_attack_enabled: bool = False   # Abilita/disabilita Eve globalmente
    eve_pns_enabled: bool = False       # Photon Number Splitting attack
    eve_pns_block_ratio: float = 0.5    # Frazione di pulse a singolo fotone da bloccare (PNS)

    # Advanced
    research_mode: bool = True       # Enable remaining key module (shares usable key for analysis)
    
    # ========== Opzionale: random attacks ==========
    use_random_attacks: bool = False    # Randomizza quali iterazioni sono attaccate
    attack_rate: float = 0.5            # Probabilità che una iterazione sia attaccata

    def validate(self):
        assert 0 <= self.interception_rate <= 1
        assert 0 <= self.attack_rate <= 1
        assert 0 <= self.depolarization_prob <= 1
        assert 0 < self.sharing_rate <= 1
        if self.use_weak_laser:
            assert 0 < self.mean_photon_num < 1
            assert self.source_frequency_hz > 0
        if self.use_channel_attenuation:
            assert self.attenuation_coeff >= 0
            assert self.distance_km >= 0
        assert 0 <= self.detector_efficiency <= 1


# ============================================================================
# Physical modules: Source, Channel, Detector
# ============================================================================

class WeakCoherentSource:
    """Simulates a weak coherent pulse source with Poisson statistics."""
    def __init__(self, mean_photon_num: float, frequency_hz: float):
        self.mu = mean_photon_num
        self.frequency = frequency_hz
        
    def generate_photons(self, required_photons: int) -> Tuple[int, float]:
        """
        Simulate generation of a pulse train until required_photons are produced.
        Returns (total_photons_generated, time_seconds).
        For each pulse, number of photons ~ Poisson(mu). Count only pulses with >=1 photon.
        """
        pulses_needed = 0
        photons_produced = 0
        while photons_produced < required_photons:
            n_photons = np.random.poisson(self.mu)
            if n_photons > 0:
                photons_produced += 1  # each pulse contributes at most one qubit (as per BB84)
            pulses_needed += 1
        time_sec = pulses_needed / self.frequency
        return photons_produced, time_sec


class Detector:
    """Models a realistic photon detector with efficiency and dead time."""
    def __init__(self, efficiency: float, dead_time_us: float = 0.0):
        self.efficiency = efficiency
        self.dead_time_sec = dead_time_us * 1e-6
        
    def detect(self, photon_arrival_times: List[float]) -> Tuple[List[bool], float]:
        """
        Simulate detection events.
        photon_arrival_times: list of absolute times when photons hit detector.
        Returns (detection_success_list, total_dead_time_penalty_sec).
        """
        detections = []
        dead_time_accum = 0.0
        last_detection_time = -np.inf
        for t in photon_arrival_times:
            if t - last_detection_time < self.dead_time_sec:
                # detector dead, no detection
                detections.append(False)
                continue
            if np.random.random() < self.efficiency:
                detections.append(True)
                last_detection_time = t
                dead_time_accum += self.dead_time_sec
            else:
                detections.append(False)
        return detections, dead_time_accum


class QuantumChannel_ImprovedPhysics:
    """
    Versione migliorata di QuantumChannel con modello fisico realistico.
    """
    
    def __init__(self,
                 amplitude_damping_gamma: float = 0.0,
                 phase_damping_lambda: float = 0.0,
                 attenuation_coeff: float = 0.0,
                 distance_km: float = 0.0,
                 depolarization_prob: float = 0.0):
        """
        Parametri fisicamente significativi:
        
        amplitude_damping_gamma: perdita di energia (0-1)
            - Per fibra ottica 10 km @ 0.2 dB/km: γ ≈ 0.15
        
        phase_damping_lambda: dephasing/decoherence (0-1)
            - Per fiber: λ ≈ 0.05
        
        depolarization_prob: depolarizzazione generica (0-1)
            - Per canale "sporco": p ≈ 0.05-0.10
        """
        self.gamma = amplitude_damping_gamma
        self.lambda_param = phase_damping_lambda
        self.p_depol = depolarization_prob
        if distance_km > 0:
            self.transmission_prob = 10 ** (-attenuation_coeff * distance_km / 10)
        else:
            self.transmission_prob = 1.0
    
    def transmit_qubit(self, state: BlochVector) -> Optional[BlochVector]:
        """
        Trasmette un singolo qubit attraverso il canale rumoroso.
        
        Parameters:
        -----------
        state : BlochVector
            Lo stato quantico preparato da Alice
        
        Returns:
        --------
        received_state : BlochVector
            Lo stato degradato dal canale
        """
        # Applica effetti del canale in sequenza
        # (ordine può importare fisicamente, ma qui trascuriamo)
        
        if np.random.random() > self.transmission_prob:
            return None  # fotone non arriva

        if self.gamma > 0:
            state.apply_amplitude_damping(self.gamma)
        
        if self.lambda_param > 0:
            state.apply_phase_damping(self.lambda_param)
        
        if self.p_depol > 0:
            state.apply_depolarization(self.p_depol)
        
        return state
 

class Alice_WithBloch:
    """
    Versione migliorata di Alice usando Bloch sphere.
    """
    
    def __init__(self, config):
        self.config = config
    
    def prepare_qubits_with_bloch(self, num_photons: int):
        """
        Prepara qubit usando Bloch sphere (fisicamente accurato).
        
        Returns:
        --------
        states : list of BlochVector
            Stati quantici preparati da Alice
        bases : np.ndarray
            Basi usate per preparare i qubit
        bits : np.ndarray
            Bit codificati
        """
        bits = np.random.randint(0, 2, size=num_photons, dtype=np.uint8)
        bases = np.random.randint(0, 2, size=num_photons, dtype=np.uint8)
        
        # NUOVO: Crea BlochVector per ogni qubit
        states = [
            BlochVector.from_basis_and_bit(bases[i], bits[i])
            for i in range(num_photons)
        ]
        
        return states, bases, bits
    

class Bob_WithBloch:
    """
    Bob con rivelatore realistico: efficiency + dead time + Born rule.
    """
    
    def __init__(self, config):
        self.config = config
        self.detector = Detector(
            efficiency=config.detector_efficiency,
            dead_time_us=config.dead_time_us if config.use_dead_time else 0.0
        )
    
    def measure_qubit(self, received_state: BlochVector, bob_basis: int, arrival_time: float = 0.0):
        """
        Misura un singolo qubit con rivelatore realistico.
        
        Prima verifica se il detector rileva il fotone (efficiency + dead time),
        poi applica Born rule sulla Bloch sphere.
        
        Returns:
        --------
        measured_bit : int (0 o 1, oppure random se non rilevato)
        detected : bool
        probability : float
        """
        detections, _ = self.detector.detect([arrival_time])
        
        if not detections[0]:
            # Fotone non rilevato: Bob assegna bit random (dark count / loss)
            return random_bit(), False, 0.5
        
        measured_bit, probability = received_state.measure(bob_basis)
        return measured_bit, True, probability
    

class BB84SimulationV2:
    """
    Versione 2 di BB84 con modello fisico Bloch sphere.

    Differenze da versione 1:
    - Usa BlochVector per stati quantici
    - Modelli di canale fisicamente accurati
    - Misurazioni con Born rule esatto
    - Tracking della purità degli stati
    """

    def __init__(self, config):
        self.config = config
        self.alice = Alice_WithBloch(config)
        self.bob = Bob_WithBloch(config)

        # Sorgente: weak coherent pulse oppure ideale
        if config.use_weak_laser:
            self.source = WeakCoherentSource(config.mean_photon_num, config.source_frequency_hz)
        else:
            self.source = None


        self._current_eve_params = {
            "interception_rate": config.interception_rate,
            "pns_enabled": config.eve_pns_enabled,
            "pns_block_ratio": config.eve_pns_block_ratio
        }
        self._attack_active_this_iteration = config.eve_attack_enabled

        # Canale con parametri fisici realistici
        self.channel = QuantumChannel_ImprovedPhysics(
            amplitude_damping_gamma=config.amplitude_damping_gamma if config.use_amplitude_damping else 0.0,
            phase_damping_lambda=config.phase_damping_lambda if config.use_phase_damping else 0.0,
            depolarization_prob=config.depolarization_prob,
            attenuation_coeff=config.attenuation_coeff if config.use_channel_attenuation else 0.0,
            distance_km=config.distance_km if config.use_channel_attenuation else 0.0,
        )
        # Storage risultati
        self.results = {
            'state_purities': [],
            'qber_estimates': [],
            'sifted_key_lengths': [],
        }


        # ========== NUOVI METODI PER CONTROLLARE EVE ==========
    def set_eve_params(self, interception_rate: float = None, 
                       pns_enabled: bool = None,
                       pns_block_ratio: float = None):
        """
        Modifica i parametri di Eve dinamicamente.
        Questo è il metodo principale che userai dal main.
        """
        if interception_rate is not None:
            self._current_eve_params["interception_rate"] = max(0.0, min(1.0, interception_rate))
        if pns_enabled is not None:
            self._current_eve_params["pns_enabled"] = pns_enabled
        if pns_block_ratio is not None:
            self._current_eve_params["pns_block_ratio"] = max(0.0, min(1.0, pns_block_ratio))
        
        # Se interception_rate > 0, attiva automaticamente l'attacco
        self._attack_active_this_iteration = (
            self._current_eve_params["interception_rate"] > 0 or 
            self._current_eve_params["pns_enabled"]
        )
    
    def disable_eve(self):
        """Disabilita completamente Eve (attacco disattivo)"""
        self.set_eve_params(interception_rate=0.0, pns_enabled=False)
    
    def enable_eve(self, interception_rate: float = 0.3):
        """Abilita Eve con interception rate specificato"""
        self.set_eve_params(interception_rate=interception_rate, pns_enabled=False)
    
    def get_eve_params(self) -> dict:
        """Restituisce i parametri correnti di Eve"""
        return self._current_eve_params.copy()
    
    def is_attack_active(self) -> bool:
        """Restituisce True se Eve è attiva nell'iterazione corrente"""
        return self._attack_active_this_iteration

    

    # ------------------------------------------------------------------
    def run(self) -> dict:
        for i in range(self.config.num_iterations):
            self.run_single_iteration_v2(i)
        return {
            'avg_qber': np.nanmean(self.results['qber_estimates']),
            'avg_sifted_len': np.mean(self.results['sifted_key_lengths']),
            'avg_purity': np.nanmean(self.results['state_purities']),
        }

    # ------------------------------------------------------------------
    def run_single_iteration_v2(self, iteration_idx: int) -> dict:
        """
        Versione 2 della simulazione con Bloch sphere.

        Fix applicati rispetto alla versione precedente:
        - Array di Bob definiti prima del loop di trasmissione
        - Loop di trasmissione e misurazione unificato (evita IndexError su fotoni persi)
        - track_state_purities rispettato correttamente
        - usable_key gestita anche quando sifted_len == 0
        """

        # Decide se questa iterazione è attaccata
        if self.config.use_random_attacks and self.config.eve_attack_enabled:
            self._attack_active_this_iteration = np.random.random() < self.config.attack_rate
        else:
            # Sincronizza con i parametri Eve correnti
            self._attack_active_this_iteration = (
                self._current_eve_params["interception_rate"] > 0 or
                self._current_eve_params["pns_enabled"]
            )

        self.source = WeakCoherentSource(
            config.mean_photon_num, config.source_frequency_hz
        ) if self.config.use_weak_laser else None

        # 0. Numero di fotoni da trasmettere
        if self.source:
            N, elapsed_time = self.source.generate_photons(self.config.raw_key_size)
        else:
            N = self.config.raw_key_size
            elapsed_time = 0.0

        # 1. Alice prepara i qubit
        alice_states, alice_bases, alice_bits = self.alice.prepare_qubits_with_bloch(N)

        # 2. Basi e array di misura di Bob (definiti PRIMA del loop)
        bob_bases = np.random.randint(0, 2, size=N, dtype=np.uint8)
        bob_bits = np.zeros(N, dtype=np.uint8)
        bob_detected = np.zeros(N, dtype=bool)
        bob_confidence = np.zeros(N, dtype=float)

        # Tempi di arrivo: 1 ns per qubit (approssimazione realistica)
        arrival_times = np.arange(N) * 1e-9

        purities = []

        # <-- NUOVO: Parametri Eve per questa iterazione
        eve_rate = self._current_eve_params["interception_rate"] if self._attack_active_this_iteration else 0.0
        eve_pns = self._current_eve_params["pns_enabled"] if self._attack_active_this_iteration else False

        # 3. Loop unico: trasmissione + misurazione qubit per qubit
        for i in range(N):

            # Stato da trasmettere (default: quello di Alice)
            current_state = alice_states[i]

            # <-- NUOVO: Eve decide se intercettare QUESTO fotone
            intercepted = False
            if self._attack_active_this_iteration and eve_rate > 0:
                if np.random.random() < eve_rate:
                    intercepted = True
                    
                    # Intercept-Resend attack con Bloch sphere
                    # Eve misura con base casuale
                    eve_basis = np.random.randint(0, 2)
                    
                    # Misura usando la funzione di Bob (riutilizziamo la logica)
                    # Ma Eve non ha detector efficiency/dead time (è ideale nell'attacco)
                    eve_bit, _, _ = self.bob.measure_qubit(
                        alice_states[i], eve_basis, arrival_time=arrival_times[i]
                    )
                    
                    if eve_basis == 0:  # Z-basis
                        if eve_bit == 0:
                            resent_state = BlochVector.z_basis_0()
                        else:
                            resent_state = BlochVector.z_basis_1()
                    else:  # X-basis
                        if eve_bit == 0:
                            resent_state = BlochVector.x_basis_plus()
                        else:
                            resent_state = BlochVector.x_basis_minus()
                    
                    # Ora il canale riceve lo stato RESO DA EVE, non quello di Alice
                    current_state = resent_state
                else:
                    current_state = alice_states[i]
            else:
                current_state = alice_states[i]
            
            # <-- NUOVO: Photon Number Splitting (se abilitato e fotone multipulse)
            if self._attack_active_this_iteration and eve_pns and self.source:
                # Per weak coherent source, alcuni pulse hanno più fotoni
                # Simuliamo il numero di fotoni nel pulse (se usiamo weak laser)
                n_photons_in_pulse = np.random.poisson(self.config.mean_photon_num)
                if n_photons_in_pulse >= 2:
                    # PNS: Eve ruba un fotone ma lascia passare il pulse
                    # Il qubit originale passa, ma Eve ha informazione
                    # Per la simulazione, non modifichiamo lo stato, ma logghiamo
                    # (opzionale: aggiungi metrica)
                    pass
                elif n_photons_in_pulse == 1 and self._current_eve_params.get("pns_block_ratio", 0) > 0:
                    # Blocca alcuni pulse a singolo fotone per nascondere la perdita
                    if np.random.random() < self._current_eve_params["pns_block_ratio"]:
                        current_state = None  # Fotone bloccato

            state_to_transmit = current_state if 'current_state' in locals() else alice_states[i]
            received_state = self.channel.transmit_qubit(state_to_transmit)

            # Fotone perso per attenuazione
            if received_state is None:
                bob_detected[i] = False
                bob_bits[i] = random_bit()
                bob_confidence[i] = 0.5
                if self.config.track_state_purities:
                    purities.append(0.5)   # stato perso → purità minima
                continue

            # Track purità (opzionale)
            if self.config.track_state_purities:
                purities.append(received_state.purity())

            # Misura di Bob
            bit, detected, confidence = self.bob.measure_qubit(
                received_state, bob_bases[i], arrival_time=arrival_times[i]
            )
            bob_bits[i] = bit
            bob_detected[i] = detected
            bob_confidence[i] = confidence

        # 4. Sifting: solo qubit rilevati E con base matchata
        match = (alice_bases == bob_bases) & bob_detected
        alice_sifted = alice_bits[match]
        bob_sifted = bob_bits[match]
        sifted_len = len(alice_sifted)

        # 5. Stima QBER
        if sifted_len > 0:
            share_size = max(1, int(sifted_len * self.config.sharing_rate))
            alice_share = alice_sifted[:share_size]
            bob_share = bob_sifted[:share_size]
            errors = np.sum(alice_share != bob_share)
            qber_est = errors / share_size
        else:
            share_size = 0
            qber_est = float('nan')

        # 6. Purità media
        avg_purity = float(np.mean(purities)) if purities else None

        # <-- NUOVO: Detection flag (se QBER supera soglia, attacco rilevato)
        detection_threshold = 0.11  # tipico per BB84
        detected = qber_est > detection_threshold if not np.isnan(qber_est) else False

        # 7. Aggiornamento storage
        self.results['state_purities'].append(avg_purity)
        self.results['qber_estimates'].append(qber_est)
        self.results['sifted_key_lengths'].append(sifted_len)

        # 8. Chiave usabile (research_mode)
        if self.config.research_mode:
            usable_key = alice_sifted[share_size:].tolist() if sifted_len > 0 else []
        else:
            usable_key = None

        return {
            'sifted_len': sifted_len,
            'qber_est': qber_est,
            'avg_purity': avg_purity,
            'usable_key': usable_key,
            'detected': detected,                    # <-- NUOVO
            'attack_active': self._attack_active_this_iteration,  # <-- NUOVO
            'eve_params': self._current_eve_params.copy() if self._attack_active_this_iteration else None
        }


    def analyze_security(self) -> Dict:
        """Restituisce analisi completa di sicurezza."""
        qber = self.results['avg_qber_est']
        
        analyzer = SecurityAnalyzer(
            n_raw=self.config.raw_key_size,
            qber=qber,
            sharing_rate=self.config.sharing_rate
        )
        
        secure_len, eve_info = analyzer.calculate_secure_key_length()
        hypothesis_test = analyzer.test_hypothesis()
        
        return {
            'secure_key_length': secure_len,
            'eve_max_information': eve_info,
            'false_positive_rate': hypothesis_test['false_positive_rate'],
            'false_negative_rate': hypothesis_test['false_negative_rate'],
            'detection_power': hypothesis_test['detection_power'],
            'shor_preskill_bound': analyzer.shor_preskill_bound(qber)
        }


if __name__ == "__main__":

    print("="*70)
    print("EXAMPLE: Integrazione Bloch Sphere nel BB84")
    print("="*70)
    
    config = SimulationConfig()

    # Esempio 1: Singolo qubit con Bloch sphere
    print("\n1. Preparazione singolo qubit:")
    print("-" * 70)
    
    state = BlochVector.from_basis_and_bit(0, 1)  # |1⟩ in Z basis
    print(f"Alice prepara |1⟩:")
    print(f"  Bloch vector: ({state.x:.3f}, {state.y:.3f}, {state.z:.3f})")
    print(f"  Purity: {state.purity():.4f}")
    
    # Applica canale
    state.apply_amplitude_damping(0.1)
    state.apply_phase_damping(0.05)
    print(f"\nDopo canale (γ=0.1, λ=0.05):")
    print(f"  Bloch vector: ({state.x:.3f}, {state.y:.3f}, {state.z:.3f})")
    print(f"  Purity: {state.purity():.4f}")
    
    # Misura
    bit, prob = state.measure(0)  # Bob misura in Z basis
    print(f"\nBob misura in Z basis:")
    print(f"  Misurato: {bit}, Probabilità: {prob:.4f}")
    
    # Esempio 2: Una semplice iterazione
    print("\n\n2. Iterazione completa con BB84 v2:")
    print("-" * 70)
    
    result = BB84SimulationV2(config).run_single_iteration_v2(0)
    
    print(f"\nRisultati:")
    print(f"  Sifted key length: {result['sifted_len']}")
    print(f"  QBER stimato: {result['qber_est']:.4f}")
    print(f"  Avg state purity: {result['avg_purity']:.4f}" if result['avg_purity'] is not None else "  Avg state purity: N/A")
    
    print("=== BB84 Single Iteration ===")
    print(f"  Sifted key length : {result['sifted_len']}")
    print(f"  QBER stimato      : {result['qber_est']:.4f}" if result['qber_est'] == result['qber_est'] else "  QBER stimato      : N/A")
    print(f"  Avg state purity  : {result['avg_purity']:.4f}" if result['avg_purity'] else "  Avg purity        : N/A")
    print(f"  Usable key bits   : {len(result['usable_key']) if result['usable_key'] is not None else 'N/A'}")


    viz = VisualizationSuite()
    viz.plot_qber_vs_noise()
    viz.plot_purity_vs_gamma()
    viz.plot_key_distribution(n_iterations=100)
    viz.plot_qber_vs_distance()
