# File: baseline_evaluation.py

from open_spiel.python.examples.bridge_wb5 import controller_factory
from absl import app
from absl import flags
import numpy as np
import pyspiel
from open_spiel.python.bots import bluechip_bridge
import time
FLAGS = flags.FLAGS
import jax.numpy as jnp
import pickle, os
import haiku as hk
import jax
import jax.numpy as jnp
import optax
from Algorithm.dummy_ai_forward import DummyNet
from Algorithm.bridge_pg_net import policy_network_fn  # 直接重用定義

def load_rl_model(step, checkpoint_dir="checkpoints/bridge_pg"):
    """載入 RL 訓練好的 Haiku 模型參數."""
    # 載入 params
    with open(os.path.join(checkpoint_dir, f"params_{step}.pkl"), "rb") as f:
        params = pickle.load(f)

    # 重建 network (需要知道 num_actions/obs_size)
    game = pyspiel.load_game("bridge(use_double_dummy_result=True)")
    num_actions = game.num_distinct_actions()
    obs_shape = game.observation_tensor_shape()
    obs_size = int(np.prod(obs_shape))

    net_fn = policy_network_fn(num_actions, hidden_units=[256, 256])
    policy_network = hk.without_apply_rng(hk.transform(net_fn))

    return policy_network, params


# 初始化
dummy_model = DummyNet()

flags.DEFINE_string("ai_model", "dummy", "選擇 AI 模型 (dummy, pg, rl2, random)")

def dummy_action(state):
    # 取得 observation (571 維)
    obs = np.array(state.observation_tensor(), dtype=np.float32)
    obs = jnp.expand_dims(obs, axis=0)  # (1, 571)

    # 模型 forward
    logits = dummy_model.forward(obs)
    action = int(jnp.argmax(logits, axis=-1)[0])  # 選 argmax 動作

    # 確保動作合法
    legal_actions = state.legal_actions()
    if action in legal_actions:
        print("argmax 合法，選擇該動作")
        return action
    else:
        print("argmax 不合法，隨機挑一個合法動作")
        return np.random.choice(legal_actions)

def ai_action_selector(state):
    model_name = FLAGS.ai_model.lower()
    
    if model_name == "dummy":
        return dummy_action(state)
    
    elif model_name == "random":
        return np.random.choice(state.legal_actions())
    
    elif model_name == "pg":
        print("使用 RL 模型 : policy gredient 選動作")
        if not hasattr(ai_action_selector, "rl_model"):
            # 第一次載入模型
            policy_network, params = load_rl_model(step=20)  # 你要選擇對應的 checkpoint
            ai_action_selector.rl_model = (policy_network, params)

        policy_network, params = ai_action_selector.rl_model

        # 準備 observation
        obs = np.array(state.observation_tensor(), dtype=np.float32)
        obs = jnp.array(obs)

        # 前向推論
        logits = policy_network.apply(params, obs)

        # 過濾不合法動作
        legal_actions_mask = jnp.array(state.legal_actions_mask())
        logits = jnp.where(legal_actions_mask, logits, -jnp.inf)

        # 選 argmax 動作
        action = int(jnp.argmax(logits))

        if action in state.legal_actions():
            print("argmax 合法，選擇該動作")
            return action
        else:
            print("argmax 不合法，隨機挑一個合法動作")
            return np.random.choice(state.legal_actions())

    
    elif model_name == "rl2":
        # TODO: 這裡放你的 RL 模型2
        return np.random.choice(state.legal_actions())  # 先佔位
    
    else:
        raise ValueError(f"未知的 ai_model: {FLAGS.ai_model}")

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
        
      if state.current_player() % 2 == 1:# 玩家 1,3
        # Have simplest play for now
        action = state.legal_actions()[0]
        if action > 51:
          # TODO(ed2k) extend beyond just bidding
            action = ai_action_selector(state)
        state.apply_action(action)
        
      else: # WBridge5 機器人 (0,2)
        result = bots[state.current_player() // 2].step(state)
        state.apply_action(result)
  return state

def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  game = pyspiel.load_game("bridge(use_double_dummy_result=false)")
  # net, params = load_model()
  net, params = None, None  # TODO: 先不使用
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