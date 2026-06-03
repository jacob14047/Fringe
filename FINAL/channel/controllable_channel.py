"""
Wrapper per BB84SimulationV2 che espone un'interfaccia semplice per il controllo di Eve.
Questo è l'unico file che gli agenti useranno per interagire con il canale.
"""

import numpy as np
from typing import Dict, Optional
from .bb84_simulator_Eve import BB84SimulationV2, SimulationConfig

class ControllableBB84:
    """
    Wrapper che aggiunge la capacità di modificare i parametri di Eve dinamicamente
    senza alterare la classe originale BB84SimulationV2.
    """
    
    def __init__(self, base_config: SimulationConfig):
        """
        Inizializza la simulazione con la configurazione base.
        
        Args:
            base_config: Configurazione della simulazione (senza parametri Eve)
        """
        self.config = base_config
        self.sim = BB84SimulationV2(base_config)
        self._current_iteration_results = None
    
    def set_eve_params(self, interception_rate: Optional[float] = None,
                       pns_enabled: Optional[bool] = None,
                       pns_block_ratio: Optional[float] = None) -> None:
        """
        Modifica i parametri di Eve per le prossime iterazioni.
        
        Args:
            interception_rate: Frazione di fotoni da intercettare (0.0 - 1.0)
            pns_enabled: Abilita Photon Number Splitting attack
            pns_block_ratio: Frazione di pulse a singolo fotone da bloccare (PNS)
        """
        self.sim.set_eve_params(
            interception_rate=interception_rate,
            pns_enabled=pns_enabled,
            pns_block_ratio=pns_block_ratio
        )
    
    def disable_eve(self) -> None:
        """Disabilita completamente Eve."""
        self.sim.disable_eve()
    
    def enable_eve(self, interception_rate: float = 0.3) -> None:
        """Abilita Eve con interception rate specificato."""
        self.sim.enable_eve(interception_rate)
    
    def get_eve_params(self) -> Dict:
        """Restituisce i parametri correnti di Eve."""
        return self.sim.get_eve_params()
    
    def run_iteration(self) -> Dict:
        """
        Esegue una singola iterazione della simulazione con i parametri Eve correnti.
        
        Returns:
            Dict con:
                - qber_est: QBER stimato
                - sifted_len: Lunghezza chiave sifted
                - avg_purity: Purità media degli stati
                - detected: True se l'attacco è stato rilevato (QBER > soglia)
                - attack_active: True se Eve era attiva in questa iterazione
                - usable_key: Chiave utilizzabile (se research_mode=True)
        """
        # La simulazione interna tiene traccia dell'indice di iterazione
        # Usiamo -1 perché non conosciamo l'indice esatto, ma la simulazione
        # gestisce internamente il contatore. Alternativa: passiamo un indice.
        # Per semplicità, chiamiamo run() che esegue TUTTE le iterazioni.
        # Ma noi vogliamo una sola iterazione. Quindi usiamo un metodo dedicato.
        
        # La tua implementazione attuale di BB84SimulationV2.run() esegue
        # num_iterations iterazioni. Dobbiamo modificarla per supportare
        # esecuzione step-by-step. Per ora, usiamo un workaround:
        
        # Salva il numero originale di iterazioni
        original_iterations = self.config.num_iterations
        self.config.num_iterations = 1
        
        # Esegui una singola iterazione
        # Nota: questo assume che run() esegua num_iterations iterazioni.
        # Se la tua run() chiama run_single_iteration_v2 in loop, funziona.
        results = self.sim.run()
        
        # Ripristina
        self.config.num_iterations = original_iterations
        
        # Estrai i risultati dell'ultima iterazione (l'unica)
        if len(self.sim.results['qber_estimates']) > 0:
            return {
                'qber_est': self.sim.results['qber_estimates'][-1],
                'sifted_len': self.sim.results['sifted_key_lengths'][-1],
                'avg_purity': self.sim.results['state_purities'][-1] if self.sim.results['state_purities'] else None,
                'detected': self.sim.results.get('detected', False),  # devi aggiungere detected ai results
                'attack_active': self.sim.is_attack_active(),
                'usable_key': None  # TODO: estrai se necessario
            }
        else:
            return {
                'qber_est': float('nan'),
                'sifted_len': 0,
                'avg_purity': None,
                'detected': False,
                'attack_active': False,
                'usable_key': None
            }
    
    def run_campaign(self, num_iterations: int) -> Dict:
        """
        Esegue una campagna di multiple iterazioni con gli stessi parametri Eve.
        
        Args:
            num_iterations: Numero di iterazioni da eseguire
            
        Returns:
            Dict con statistiche aggregate
        """
        qbers = []
        sifted_lens = []
        detections = []
        
        for _ in range(num_iterations):
            result = self.run_iteration()
            qbers.append(result['qber_est'])
            sifted_lens.append(result['sifted_len'])
            detections.append(result['detected'])
        
        return {
            'avg_qber': np.nanmean(qbers),
            'std_qber': np.nanstd(qbers),
            'avg_sifted_len': np.mean(sifted_lens),
            'detection_rate': np.mean(detections),
            'all_qbers': qbers,
            'all_sifted_lens': sifted_lens
        }