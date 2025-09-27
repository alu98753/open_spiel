import pyspiel
from open_spiel.python import rl_environment

game = pyspiel.load_game("bridge")  # 或 "uncontested_bridge_bidding", "tiny_bridge"
env = rl_environment.Environment(game)
ts = env.reset()
pid = ts.observations["current_player"]
info = ts.observations["info_state"][pid]
legal = ts.observations["legal_actions"][pid]
print("info_state length =", len(info))
print("sample of info_state (first 32 dims) =", info[:32])
print("legal_actions =", legal)
