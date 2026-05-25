import json
import re
from channel.controllable_channel import ControllableBB84
from channel.bb84_simulator_Eve import SimulationConfig
from tracking.mlflow_logger import MLflowLogger
from agents.planning_agent import create_simple_planner

def _parse_llm_json(text):
    """Estrae un JSON valido dal testo dell'LLM."""
    # Prova prima il testo intero
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Prova a estrarre da code block ```json ... ```
    for m in re.finditer(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', text):
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Cerca pattern "interception_rate": numero
    for m in re.finditer(r'"interception_rate"\s*:\s*([\d.]+)[^}]*?"pns_enabled"\s*:\s*(true|false)', text, re.IGNORECASE):
        try:
            return {
                "interception_rate": float(m.group(1)),
                "pns_enabled": m.group(2).lower() == "true"
            }
        except (ValueError, IndexError):
            pass
    # Cerca pattern inverso
    for m in re.finditer(r'"pns_enabled"\s*:\s*(true|false)[^}]*?"interception_rate"\s*:\s*([\d.]+)', text, re.IGNORECASE):
        try:
            return {
                "pns_enabled": m.group(1).lower() == "true",
                "interception_rate": float(m.group(2))
            }
        except (ValueError, IndexError):
            pass
    # Fallback
    print(f"[WARN] JSON non trovato. LLM returned: {text[:120]}...")
    return {"interception_rate": 0.2, "pns_enabled": False}

def main():
    config = SimulationConfig()
    channel = ControllableBB84(config)
    planner = create_simple_planner()
    logger = MLflowLogger(experiment_name="single_agent_test")
    
    for episode in range(5):
        # 1. Esegui canale senza Eve per ottenere baseline
        channel.set_eve_params(interception_rate=0.0)
        baseline = channel.run_iteration()
        
        # 2. Chiedi all'agente di decidere l'attacco
        response = planner.invoke({"qber": baseline["qber_est"], "sifted_len": baseline["sifted_len"]})
        params = _parse_llm_json(response)
        
        # 3. Esegui con l'attacco
        channel.set_eve_params(**params)
        attack_result = channel.run_iteration()
        
        with logger.start_run(run_name=f"episode_{episode}"):
            logger.log_params(params)
            logger.log_metrics({
                "baseline_qber": baseline["qber_est"],
                "attack_qber": attack_result["qber_est"],
                "attack_success": 1 if attack_result["detected"] is False else 0
            })
        print(f"Ep {episode}: {params} -> QBER={attack_result['qber_est']:.4f}")