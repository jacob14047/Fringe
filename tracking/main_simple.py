from channel.controllable_channel import ControllableBB84
from tracking.mlflow_logger import MLflowLogger
from channel.bb84_simulator_Eve import SimulationConfig

def main():
    config = SimulationConfig(raw_key_size=4000, num_iterations=1)
    channel = ControllableBB84(config)
    logger = MLflowLogger(experiment_name="baseline_no_agents")
    
    for rate in [0.0, 0.1, 0.3, 0.5]:
        with logger.start_run(run_name=f"eve_{rate}"):
            channel.set_eve_params(interception_rate=rate)
            result = channel.run_iteration()
            logger.log_params({"interception_rate": rate})
            logger.log_metrics({"qber": result["qber_est"], "sifted_len": result["sifted_len"]})
            print(f"Rate={rate}: QBER={result['qber_est']:.4f}")

if __name__ == "__main__":
    main()