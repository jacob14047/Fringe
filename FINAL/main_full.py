import json
from channel.controllable_channel import ControllableBB84
from channel.bb84_simulator_Eve import SimulationConfig
from tracking.mlflow_logger import MLflowLogger
from bayesian.Dempster_Shafer import DempsterShafer
from agents.recon_agent import get_recon_bpa
from agents.planning_agent import create_advanced_planner
from agents.execution_agent import get_execution_bpa  

def main():
    config = SimulationConfig(raw_key_size=4000, num_iterations=1)
    channel = ControllableBB84(config)
    planner = create_advanced_planner()
    ds = DempsterShafer(["Vulnerabile", "Non Vulnerabile"])
    logger = MLflowLogger(experiment_name="dempster_shafer_full")
    
    # Valori iniziali per belief e plaus (prima del primo aggiornamento)
    belief = 0.5
    plaus = 0.5

    for episode in range(30):
        print(f"\n🔵 Episodio {episode} - Recon")
        
        # 1. Baseline senza attacco (Recon)
        channel.set_eve_params(interception_rate=0.0)
        baseline = channel.run_iteration()
        
        qber_base = baseline.get("qber_est", baseline.get("qber", 0.0))
        sifted_len_base = baseline.get("sifted_len", 0)
        print(f"   Baseline QBER={qber_base:.4f}, len={sifted_len_base}")
        
        # 2. BPA dal Recon
        bpa_recon = get_recon_bpa(qber_base, sifted_len_base)
        ds.assign_mass(bpa_recon, source_name="Recon")
        
        # 3. Calcola belief e plausibility DOPO aver assegnato la massa
        belief = ds.get_belief(["Vulnerabile"])
        plaus = ds.get_plausibility(["Vulnerabile"])
        print(f"   Belief={belief:.3f}, Plaus={plaus:.3f}")
        
        # 2. Planning (con timeout e retry)
        attack_params = {"interception_rate": 0.2, "pns_enabled": False}
        try:
            print("   Planning: invocazione LLM...")
            response = planner.invoke({
                "belief": belief,
                "plausibility": plaus,
                "qber": qber_base
            })
            print(f"   LLM risponde: {response[:200]}")
            attack_params = json.loads(response)
        except Exception as e:
            print(f"   ❌ Errore LLM: {e}")
            # fallback: aumenta gradualmente rate se belief alta
            if belief > 0.6:
                attack_params = {"interception_rate": 0.25, "pns_enabled": False}
            elif belief > 0.3:
                attack_params = {"interception_rate": 0.2, "pns_enabled": False}
            else:
                attack_params = {"interception_rate": 0.1, "pns_enabled": False}
        
        # 3. Execution
        print(f"   Esecuzione attacco: {attack_params}")
        channel.set_eve_params(
            interception_rate=attack_params.get("interception_rate", 0.0),
            pns_enabled=attack_params.get("pns_enabled", False)
        )
        attack_result = channel.run_iteration()
        detected = attack_result.get("detected", False)
        qber_attack = attack_result.get("qber_est", attack_result.get("qber", 0.0))
        
        bpa_exec = get_execution_bpa(detected)
        ds.assign_mass(bpa_exec, source_name="Execution")
        
        # Nuovi belief e plaus dopo execution
        belief = ds.get_belief(["Vulnerabile"])
        plaus = ds.get_plausibility(["Vulnerabile"])
        
        # 4. Logging
        with logger.start_run(run_name=f"ep_{episode}"):
            logger.log_params(attack_params)
            logger.log_metrics({
                "baseline_qber": qber_base,
                "attack_qber": qber_attack,
                "attack_success": 1 if not detected else 0,
                "belief_vulnerable": belief,
                "plausibility_vulnerable": plaus
            })
        
        print(f"Ep{episode}: belief={belief:.3f} -> attack {attack_params} success={not detected}")


if __name__ == "__main__":
    main()