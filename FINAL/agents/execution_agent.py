# src/agents/execution_agent.py
import json
from channel.controllable_channel import ControllableBB84

class ExecutionAgent:
    def __init__(self, channel: ControllableBB84):  # ← wrapper, non simulazione diretta
        self.channel = channel
    
    def execute(self, strategy_json: str) -> dict:
        params = json.loads(strategy_json)
        self.channel.set_eve_params(
            interception_rate=params.get("interception_rate", 0.0),
            pns_enabled=params.get("pns_enabled", False)  # ← chiave corretta
        )
        result = self.channel.run_iteration()  # ← restituisce dict con 'detected'
        return {
            "success": not result.get("detected", False),  # ← attacco riuscito se NON rilevato
            "qber": result.get("qber_est", float('nan')),
            "sifted_len": result.get("sifted_len", 0)
        }
    
    
def get_execution_bpa(detected: bool) -> dict:
    if detected:
        # Attacco rilevato -> evidenza di non vulnerabilità
        return {frozenset(["Non Vulnerabile"]): 0.6, frozenset(["Vulnerabile"]): 0.1, frozenset(["Vulnerabile", "Non Vulnerabile"]): 0.3}
    else:
        # Attacco non rilevato -> evidenza di vulnerabilità
        return {frozenset(["Vulnerabile"]): 0.7, frozenset(["Non Vulnerabile"]): 0.05, frozenset(["Vulnerabile", "Non Vulnerabile"]): 0.25}