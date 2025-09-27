# train_model.py
import haiku as hk
import jax
import numpy as np
import pickle
import os

# 定義與 bridge_wb5.py 中相同的網路結構
NUM_ACTIONS = 38

def net_fn(x):
  net = hk.Sequential([
      hk.Linear(1024),
      jax.nn.relu,
      hk.Linear(1024),
      jax.nn.relu,
      hk.Linear(1024),
      jax.nn.relu,
      hk.Linear(1024),
      jax.nn.relu,
      hk.Linear(NUM_ACTIONS),
      jax.nn.log_softmax,
  ])
  return net(x)

def main():
  # 初始化網路
  net = hk.without_apply_rng(hk.transform(net_fn))

  # 為了得到參數，我們需要一個虛擬的輸入
  dummy_input = np.zeros(shape=(1, 106), dtype=np.float32) 
  # bridge 的 observation_tensor 大小是 106

  # 初始化參數
  rng = jax.random.PRNGKey(42)
  params = net.init(rng, dummy_input)

  # 定義儲存路徑
  output_dir = "/home/asiadragon/Desktop/open_spiel/open_spiel/python/examples/mytest0902/" # 或者你想要的任何路徑
  output_file = os.path.join(output_dir, "params-snapshot.pkl")

  # 儲存參數
  with open(output_file, "wb") as f:
    pickle.dump(params, f)

  print(f"模型參數已儲存至 {output_file}")

if __name__ == "__main__":
  main()