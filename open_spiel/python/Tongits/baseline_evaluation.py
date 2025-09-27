# File: baseline_evaluation.py

from open_spiel.python.examples.bridge_wb5 import controller_factory
from open_spiel.python.examples.bridge_wb5 import load_model
from absl import app
from absl import flags
import numpy as np
from open_spiel.python.examples.bridge_wb5 import ai_action
import pyspiel
from open_spiel.python.bots import bluechip_bridge
import time
FLAGS = flags.FLAGS

def _run_once(state, bots, net, params):
  """Plays bots with each other, returns terminal utility for each player."""
  for bot in bots:
    bot.restart()
  while not state.is_terminal():
    if state.is_chance_node():
      outcomes, probs = zip(*state.chance_outcomes())
      state.apply_action(np.random.choice(outcomes, p=probs))
    else:
      if FLAGS.sleep:
        time.sleep(FLAGS.sleep)  # wait for the human to see how it goes
      if state.current_player() % 2 == 1:
        # Have simplest play for now
        action = state.legal_actions()[0]
        if action > 51:
          # TODO(ed2k) extend beyond just bidding
          action = ai_action(state, net, params)
        state.apply_action(action)
      else:
        result = bots[state.current_player() // 2].step(state)
        state.apply_action(result)
  return state

def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  game = pyspiel.load_game("bridge(use_double_dummy_result=false)")
  net, params = load_model()
  bots = [
      bluechip_bridge.BlueChipBridgeBot(game, 0, controller_factory),
      bluechip_bridge.BlueChipBridgeBot(game, 2, controller_factory)
  ]

  results = []

  for i_deal in range(FLAGS.num_deals):
    state =  _run_once(game.new_initial_state(), bots, net, params)
    print("Deal #{}; final state:\n{}".format(i_deal, state))
    print(f"  完成對局: {i_deal + 1}/{FLAGS.num_deals}", end='\r')

    results.append(state.returns())

  stats = np.array(results)
  # mean = np.mean(stats, axis=0)
  # stderr = np.std(stats, axis=0, ddof=1) / np.sqrt(FLAGS.num_deals)
  # print(u"Absolute score: {:+.1f}\u00b1{:.1f}".format(mean[0], stderr[0]))
  # print(u"Relative score: {:+.1f}\u00b1{:.1f}".format(mean[1], stderr[1]))
  output_report(stats)

def output_report(stats):
  # WBridge5 隊伍: 北南 (0, 2)
  ns_scores = stats[:, 0] + stats[:, 2] 
  # UniformRandom 隊伍: 東西 (1, 3)
  ew_scores = stats[:, 1] + stats[:, 3] 

  # 計算 WBridge5 隊相對於 UniformRandom 隊的平均得分差異
  score_diffs = ns_scores - ew_scores
  
  mean_diff = np.mean(score_diffs)
  # 計算標準誤 (Standard Error)
  std_err_diff = np.std(score_diffs, ddof=1) / np.sqrt(FLAGS.num_deals)
  
  print("\n--- 基準測試最終結果 ---")
  print(f"對局總數: {FLAGS.num_deals}")
  print(f"WBridge5 (NS) 平均得分: {np.mean(ns_scores):.2f}")
  print(f"UniformRandom (EW) 平均得分: {np.mean(ew_scores):.2f}")
  print(u"平均得分差異 (WBridge5 - UniformRandom): {:+.2f} ± {:.2f}".format(mean_diff, std_err_diff))
  
  # 勝率計算: 假設 NS 隊得分差異 > 0 算作勝利
  ns_wins = np.sum(score_diffs > 0)
  ew_wins = np.sum(score_diffs < 0)
  draws = np.sum(score_diffs == 0)
  
  ns_win_rate = ns_wins / FLAGS.num_deals
  
  print(f"WBridge5 (NS) 勝率: {ns_win_rate:.2%}")
  print(f"UniformRandom (EW) 勝率: {ew_wins / FLAGS.num_deals:.2%}")
  print(f"平局率: {draws / FLAGS.num_deals:.2%}")  

if __name__ == "__main__":
  app.run(main)