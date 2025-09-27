import haiku as hk
import jax
import jax.numpy as jnp

INPUT_DIM = 571     # 觀測維度
NUM_ACTIONS = 38    # 橋牌動作空間大小

class DummyNet:
    def __init__(self, rng_seed=42):
        # 包裝成 Haiku 模組
        self.net = hk.without_apply_rng(hk.transform(self._net_fn))
        self.rng = jax.random.PRNGKey(rng_seed)

        # 初始化參數
        sample_input = jnp.zeros([1, INPUT_DIM])   # batch=1, dim=571
        self.params = self.net.init(self.rng, sample_input)

    def _net_fn(self, x):
        """簡單的 Dummy MLP 模型"""
        net = hk.Sequential([
            hk.Linear(128),   # 隱藏層
            jax.nn.relu,
            hk.Linear(NUM_ACTIONS)  # 輸出動作 logits
        ])
        return net(x)

    def forward(self, inputs):
        """前向傳播，輸入觀測向量 (batch_size, 571)，輸出 logits"""
        return self.net.apply(self.params, inputs)


if __name__ == "__main__":
    model = DummyNet()
    sample_input = jnp.zeros([1, INPUT_DIM])
    logits = model.forward(sample_input)
    print("輸出 logits shape:", logits.shape)  # (1, 38)
