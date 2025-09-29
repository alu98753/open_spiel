# File: baseline_evaluation.py

from open_spiel.python.examples.bridge_wb5 import controller_factory
from absl import app
from absl import flags
import numpy as np
import pyspiel
from open_spiel.python.bots import bluechip_bridge
import time
import jax.numpy as jnp
import pickle, os
import haiku as hk
import jax
import jax.numpy as jnp
import optax
from Algorithm.dummy_ai_forward import DummyNet
from open_spiel.python.Tongits.Algorithm.bridge_pg_trainer import policy_network_fn  # 直接重用定義

FLAGS = flags.FLAGS
flags.DEFINE_string("ns_model", "dummy", "NS 隊 (玩家 0,2) 使用的 AI 模型")
flags.DEFINE_string("ew_model", "random", "EW 隊 (玩家 1,3) 使用的 AI 模型")

# load policy gradient model
def load_pg_model(step, checkpoint_dir= os.path.join(  os.path.dirname(os.path.abspath(__file__)), "checkpoints/bridge_pg")):
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

### nfsp model loading function (if needed)
from open_spiel.python.jax import nfsp
from open_spiel.python import rl_environment
def load_nfsp_agents(checkpoint_dir, num_players=4):
    """從 checkpoint 載入 NFSP agents。"""
    game = "bridge(use_double_dummy_result=true)"
    env = rl_environment.Environment(game)
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    agents = []
    for pid in range(num_players):
        agent = nfsp.NFSP(
            player_id=pid,
            state_representation_size=info_state_size,
            num_actions=num_actions,
            hidden_layers_sizes=[256, 256],
            reservoir_buffer_capacity=200000,
            anticipatory_param=0.1,
        )
        ckpt_dir = os.path.join(checkpoint_dir, f"agent{pid}")
        if os.path.exists(ckpt_dir):
            agent.restore(ckpt_dir)   # restore 從該 agent 的資料夾
            print(f"載入 NFSP agent {pid} 成功 (路徑={ckpt_dir})")
        else:
            print(f"⚠️ 找不到 {ckpt_dir}，使用隨機初始化")
        agents.append(agent)
    return agents



# load dummy model
dummy_model = DummyNet()
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
        # print("argmax 合法，選擇該動作")
        return action
    else:
        # print("argmax 不合法，隨機挑一個合法動作")
        return np.random.choice(legal_actions)


def ai_action_selector(state, model_name):
    """根據 model_name 選擇不同的 AI 行為"""
    model_name = model_name.lower()
    
    if model_name == "dummy":
        return dummy_action(state)
    
    elif model_name == "random":
        return np.random.choice(state.legal_actions())
    
    elif model_name == "pg":
        print("使用 RL 模型 : policy gradient 選動作")
        if not hasattr(ai_action_selector, "rl_model"):
            policy_network, params = load_pg_model(step=100000)  # TODO: 修改 checkpoint step
            ai_action_selector.rl_model = (policy_network, params)

        policy_network, params = ai_action_selector.rl_model
        obs = np.array(state.observation_tensor(), dtype=np.float32)
        obs = jnp.array(obs)

        logits = policy_network.apply(params, obs)
        legal_actions_mask = jnp.array(state.legal_actions_mask())
        logits = jnp.where(legal_actions_mask, logits, -jnp.inf)

        action = int(jnp.argmax(logits))
        if action in state.legal_actions():
            return action
        else:
            return np.random.choice(state.legal_actions())

    elif model_name == "nfsp":
        if not hasattr(ai_action_selector, "nfsp_agents"):
            checkpoint_dir = os.path.join(  os.path.dirname(os.path.abspath(__file__)), "checkpoints/bridge_nfsp") 
            ai_action_selector.nfsp_agents = load_nfsp_agents(checkpoint_dir)
        cur = state.current_player()
        agent = ai_action_selector.nfsp_agents[cur]

        obs = {
            "current_player": cur,
            "info_state": [None] * 4,
            "legal_actions": [None] * 4
        }
        obs["info_state"][cur] = state.information_state_tensor(cur)
        obs["legal_actions"][cur] = state.legal_actions(cur)

        time_step = rl_environment.TimeStep(observations=obs, rewards=None,
                                            discounts=None, step_type=None)
        step_output = agent.step(time_step, is_evaluation=True)
        action = step_output.action

        if action in state.legal_actions():
            return action
        else:
            return np.random.choice(state.legal_actions())


    else:
        raise ValueError(f"未知的 ai_model: {model_name}")

def _run_once(state, net, params):
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes, probs = zip(*state.chance_outcomes())
            state.apply_action(np.random.choice(outcomes, p=probs))
        else:
            if FLAGS.sleep:
                time.sleep(FLAGS.sleep)

            cur = state.current_player()
            if cur in [0, 2]:   # NS 隊
                action = ai_action_selector(state, FLAGS.ns_model)
            else:               # EW 隊
                action = ai_action_selector(state, FLAGS.ew_model)
            state.apply_action(action)

    return state


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")
    game = pyspiel.load_game("bridge(use_double_dummy_result=false)")
    # net, params = load_model()
    net, params = None, None  # TODO: 先不使用


    results = []
    start_time = time.time()
    for i_deal in range(FLAGS.num_deals):
        state =  _run_once(game.new_initial_state(), net, params)
        print("Deal #{}; final state:\n{}".format(i_deal, state))
        print(f"  完成對局: {i_deal + 1}/{FLAGS.num_deals}", end='\r')

        results.append(state.returns())
        
    print("Cost time: {:.2f} 秒".format(time.time() - start_time))
    stats = np.array(results)
    # mean = np.mean(stats, axis=0)
    # stderr = np.std(stats, axis=0, ddof=1) / np.sqrt(FLAGS.num_deals)
    # print(u"Absolute score: {:+.1f}\u00b1{:.1f}".format(mean[0], stderr[0]))
    # print(u"Relative score: {:+.1f}\u00b1{:.1f}".format(mean[1], stderr[1]))
    output_report(stats)

def output_report(stats):
    # NS 隊伍: 玩家 0 + 2
    ns_scores = stats[:, 0] + stats[:, 2]
    # EW 隊伍: 玩家 1 + 3
    ew_scores = stats[:, 1] + stats[:, 3]

    score_diffs = ns_scores - ew_scores
    mean_diff = np.mean(score_diffs)
    std_err_diff = np.std(score_diffs, ddof=1) / np.sqrt(FLAGS.num_deals)

    ns_name = FLAGS.ns_model.upper()
    ew_name = FLAGS.ew_model.upper()

    print("\n--- 基準測試最終結果 ---")
    print(f"對局總數: {FLAGS.num_deals}")
    print(f"{ns_name} (NS) 平均得分: {np.mean(ns_scores):.2f}")
    print(f"{ew_name} (EW) 平均得分: {np.mean(ew_scores):.2f}")
    print(u"平均得分差異 ({} - {}): {:+.2f} ± {:.2f}".format(ns_name, ew_name, mean_diff, std_err_diff))

    ns_wins = np.sum(score_diffs > 0)
    ew_wins = np.sum(score_diffs < 0)
    draws = np.sum(score_diffs == 0)
    ns_win_rate = ns_wins / FLAGS.num_deals

    print(f"{ns_name} (NS) 勝率: {ns_win_rate:.2%}")
    print(f"{ew_name} (EW) 勝率: {ew_wins / FLAGS.num_deals:.2%}")
    print(f"平局率: {draws / FLAGS.num_deals:.2%}")
    print(f"NS 勝 {ns_wins} 局, EW 勝 {ew_wins} 局, 平局 {draws} 局")
    

if __name__ == "__main__":
  app.run(main)