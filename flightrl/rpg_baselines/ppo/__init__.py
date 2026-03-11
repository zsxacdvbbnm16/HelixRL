# PPO implementations for RPG Baselines
from .ppo2 import PPO2
from .ppo2_per import PPO2withPER
# from .ppo2_sil import PPO2_SIL  # Commented out - not needed for federated learning
# from .ppo2_test import PPO2_TEST  # Commented out - not needed for federated learning

# Optional federated/privacy extension (depends on extra packages like sklearn)
try:
    from .federated_ppo import FederatedPPO2, create_federated_ppo_client
except Exception:
    pass
