from channel.bb84_simulator_Eve import SimulationConfig
from channel.controllable_channel import ControllableBB84

def test():
    # Configurazione base (senza Eve statica)
    config = SimulationConfig(
        raw_key_size=1000,
        num_iterations=1,
        depolarization_prob=0.02,
        use_weak_laser=False,  # Sorgente ideale per test
        research_mode=False
    )
    
    channel = ControllableBB84(config)
    
    print("=== Test 1: Senza Eve ===")
    channel.disable_eve()
    res1 = channel.run_iteration()
    print(f"QBER: {res1['qber_est']:.4f}, Detected: {res1['detected']}")
    
    print("\n=== Test 2: Con Eve (30% intercept) ===")
    channel.enable_eve(interception_rate=0.3)
    res2 = channel.run_iteration()
    print(f"QBER: {res2['qber_est']:.4f}, Detected: {res2['detected']}")
    
    print("\n=== Test 3: Campagna 10 iterazioni ===")
    channel.disable_eve()
    campaign = channel.run_campaign(10)
    print(f"Avg QBER: {campaign['avg_qber']:.4f}, Detection rate: {campaign['detection_rate']:.2f}")

if __name__ == "__main__":
    test()