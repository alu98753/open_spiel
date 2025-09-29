# Copyright 2025 OpenSpiel 專家
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
為 OpenSpiel 橋牌遊戲實現 Policy Gradient (REINFORCE) 演算法的訓練腳本。

此腳本遵循以下核心邏輯：
1.  **Self-Play**: 讓四個玩家使用一個共享權重的策略網路進行對局。
2.  **數據收集**: 記錄下南北 (NS) 和東西 (EW) 兩個隊伍各自的決策軌跡。
3.  **獎勵計算**: 在叫牌結束後，利用 OpenSpiel 內建的雙明手分析器獲得最終得分。
4.  **模型更新**: 使用 REINFORCE 演算法，根據最終得分來更新共享的策略網路。
"""

from absl import app
from absl import flags
from absl import logging

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyspiel
from functools import partial
import os, pickle

FLAGS = flags.FLAGS

# flags.DEFINE_integer("num_deals", 100000, "要訓練的總牌局數。")
# flags.DEFINE_float("learning_rate", 1e-4, "優化器的學習率。")
# flags.DEFINE_integer("print_every", 1000, "每隔多少牌局打印一次訓練狀態。")
# flags.DEFINE_string("checkpoint_dir", f"/mnt/zi/Master_Thesis/src/open_spiel/open_spiel/python/Tongits/checkpoints/bridge_pg", "模型存檔路徑")
# flags.DEFINE_integer("save_every", 10000, "每隔多少牌局存一次 checkpoint")
# flags.DEFINE_integer("load_step", 0, "若大於 0，從指定步數的 checkpoint 載入模型")


def save_checkpoint(params, optimizer_state, step, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/params_{step}.pkl", "wb") as f:
        pickle.dump(params, f)
    with open(f"{save_dir}/optimizer_{step}.pkl", "wb") as f:
        pickle.dump(optimizer_state, f)
    logging.info(f"Checkpoint saved at step {step}")

def load_checkpoint(step, save_dir):
    with open(f"{save_dir}/params_{step}.pkl", "rb") as f:
        params = pickle.load(f)
    with open(f"{save_dir}/optimizer_{step}.pkl", "rb") as f:
        optimizer_state = pickle.load(f)
    logging.info(f"Loaded checkpoint from step {step}")
    return params, optimizer_state


# --- 1. 定義神經網路模型 ---
# 使用 Haiku 建立一個簡單的多層感知器 (MLP) 作為我們的策略網路。
# 它的任務是接收遊戲狀態的觀察值 (observation)，並輸出每個可能動作的機率分佈。
def policy_network_fn(num_actions: int, hidden_units: list[int]):
  """Haiku MLP 函數."""
  def forward(obs):
    layers = []
    for num_units in hidden_units:
      layers.extend([
          hk.Linear(num_units),
          jax.nn.relu,
      ])
    # 最後一層輸出 logits，維度等於動作空間大小
    layers.append(hk.Linear(num_actions))
    mlp = hk.Sequential(layers)
    return mlp(obs)
  return forward


# --- 2. REINFORCE 演算法核心：損失函數與更新步驟 ---
# 這是演算法的核心。我們將它封裝在一個函數中，並用 @jax.jit 加速。
# 使用 static_argnames 告訴 JAX，'net_apply' 是一個靜態參數（一個 Python 函數），不應被追蹤。
@partial(jax.jit, static_argnames=("net_apply",))
def train_step(params, optimizer_state, net_apply, ns_trajectory, ew_trajectory, ns_return, ew_return):
    """
    執行一次完整的梯度計算與模型更新。
    這個函數會被 JAX JIT 編譯以提高效能。
    """
  
    def loss_fn(p, trajectory, final_return):
        """
        計算單一隊伍的 REINFORCE 損失。
        損失 = -log(π(a|s)) * G
        其中 π 是策略，a 是採取的動作，s 是狀態，G 是最終的總獎勵。
        """
        # 將 Python 列表轉換為 JAX 可處理的元組
        trajectory = tuple(trajectory)
        total_loss = 0.0
        for transition in trajectory:
            obs, action = transition
            # 將 observation 輸入網路，得到所有動作的 logits
            logits = net_apply(p, obs)
            # 使用 log_softmax 將 logits 轉換為 log probabilities
            log_probs = jax.nn.log_softmax(logits)
            # 取出實際採取動作的 log probability
            log_prob_action = log_probs[action]
            # REINFORCE 損失，負號是為了執行梯度上升 (最大化獎勵)
            total_loss -= log_prob_action * final_return
            # 回傳平均損失
        return total_loss / len(trajectory)

    # 計算梯度
    ns_loss, ns_grads = jax.value_and_grad(loss_fn)(params, ns_trajectory, ns_return)
    ew_loss, ew_grads = jax.value_and_grad(loss_fn)(params, ew_trajectory, ew_return)
    
    # 將兩個隊伍的梯度合併（因為是同一個網路）
    total_grads = jax.tree_util.tree_map(lambda x, y: x + y, ns_grads, ew_grads)
    
    # 使用 Optax 更新模型參數
    updates, new_optimizer_state = optimizer.update(total_grads, optimizer_state)
    new_params = optax.apply_updates(params, updates)
    
    total_loss = ns_loss + ew_loss
    
    return new_params, new_optimizer_state, total_loss

def main(_):
    # --- 3. 初始化遊戲、模型與優化器 ---
    logging.info("正在載入橋牌遊戲...")
    # 關鍵參數：`use_double_dummy_result=True` 讓遊戲在結束時直接回傳雙明手分析結果
    game = pyspiel.load_game("bridge(use_double_dummy_result=True)")
    
    num_actions = game.num_distinct_actions()
    obs_shape = game.observation_tensor_shape()
    # 橋牌的 observation 是一個扁平化的向量
    obs_size = np.prod(obs_shape) 

    # 將模型函數轉換為 Haiku 的標準格式
    net_fn = policy_network_fn(num_actions, hidden_units=[256, 256])
    policy_network = hk.without_apply_rng(hk.transform(net_fn))

    # 初始化模型參數
    rng_key = jax.random.PRNGKey(42)
    dummy_obs = jnp.zeros((obs_size,))
    params = policy_network.init(rng_key, dummy_obs)

    # 初始化 Adam 優化器
    global optimizer
    optimizer = optax.adam(FLAGS.learning_rate)
    optimizer_state = optimizer.init(params)
    
    # 如果指定了 load_step，則從 checkpoint 載入模型
    if FLAGS.load_step > 0:
        params, optimizer_state = load_checkpoint(FLAGS.load_step, FLAGS.checkpoint_dir)

    
    logging.info("初始化完成，開始訓練...")
    
    # --- 4. 主訓練迴圈 ---
    for deal_num in range(1, FLAGS.num_deals + 1):
        # a. Self-Play & 數據收集
        state = game.new_initial_state()
        ns_trajectory = []  # (obs, action)
        ew_trajectory = []  # (obs, action)
        print("第 {} 局".format(deal_num))
        while not state.is_terminal():
            if state.is_chance_node():
                # 處理發牌
                outcomes, _ = zip(*state.chance_outcomes())
                action = np.random.choice(outcomes)
                state.apply_action(action)
                continue

            current_player = state.current_player()
            
            # 獲取當前玩家的觀察
            obs = jnp.array(state.observation_tensor(current_player))
            
            # 使用策略網路選擇動作
            logits = policy_network.apply(params, obs)
            
            # 過濾掉不合法的動作
            legal_actions_mask = jnp.array(state.legal_actions_mask(current_player))
            logits = jnp.where(legal_actions_mask, logits, -jnp.inf)
            
            # 從策略分佈中採樣一個動作
            # 注意：JAX 的 random function 需要一個 PRNGKey
            rng_key, action_key = jax.random.split(rng_key)
            action = jax.random.categorical(action_key, logits)
            action = int(action) # 轉為 Python int
            
            # 記錄這次決策
            if current_player in [0, 2]: # North/South
                ns_trajectory.append((obs, action))
            else: # East/West
                ew_trajectory.append((obs, action))
                
            # 在遊戲中執行動作
            state.apply_action(action)
        
        # b. 獲取最終獎勵
        returns = state.returns()
        ns_return = returns[0]
        ew_return = returns[1]
        
        # c. 執行學習步驟 (如果該局有產生決策)
        if ns_trajectory and ew_trajectory:
            params, optimizer_state, loss = train_step(
                params, optimizer_state, policy_network.apply, 
                ns_trajectory, ew_trajectory, ns_return, ew_return
            )
            
        # d. 打印日誌
        if deal_num % FLAGS.print_every == 0:
            logging.info(f"Deal #{deal_num}, Loss: {loss:.4f}, NS Return: {ns_return}, EW Return: {ew_return}")
        if deal_num % FLAGS.save_every == 0:
            save_checkpoint(params, optimizer_state, deal_num, FLAGS.checkpoint_dir)
if __name__ == "__main__":
    app.run(main)