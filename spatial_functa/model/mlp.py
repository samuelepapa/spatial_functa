import jax.numpy as jnp
from flax import linen as nn


class MLP(nn.Module):
    hidden_dim: int
    num_layers: int
    num_classes: int
    dropout_prob: float = 0.0

    @nn.compact
    def __call__(self, x, train=True):
        for l in range(self.num_layers):
            x = nn.Dense(self.hidden_dim)(x)
            # x = nn.LayerNorm()(x)
            x = nn.gelu(x)

        x = nn.Dense(self.num_classes)(x)

        return x
