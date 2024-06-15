import flax.linen as nn
import jax.numpy as jnp


class AttentionBlock(nn.Module):
    embed_dim: int  # Dimensionality of input and attention feature vectors
    hidden_dim: int  # Dimensionality of hidden layer in feed-forward network
    num_heads: int  # Number of heads to use in the Multi-Head Attention block
    dropout_prob: float = 0.0  # Amount of dropout to apply in the feed-forward network

    def setup(self):
        self.attn = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)
        self.linear = [
            nn.Dense(self.hidden_dim),
            nn.gelu,
            nn.Dropout(self.dropout_prob),
            nn.Dense(self.embed_dim),
        ]
        self.layer_norm_1 = nn.LayerNorm()
        self.layer_norm_2 = nn.LayerNorm()
        self.dropout = nn.Dropout(self.dropout_prob)

    def __call__(self, x, train=True):
        inp_x = self.layer_norm_1(x)
        attn_out = self.attn(inputs_q=inp_x, inputs_kv=inp_x)
        x = x + self.dropout(attn_out, deterministic=not train)

        linear_out = self.layer_norm_2(x)
        for l in self.linear:
            linear_out = (
                l(linear_out)
                if not isinstance(l, nn.Dropout)
                else l(linear_out, deterministic=not train)
            )
        x = x + self.dropout(linear_out, deterministic=not train)
        return x


class VisionTransformer(nn.Module):
    embed_dim: int  # Dimensionality of input and attention feature vectors
    hidden_dim: int  # Dimensionality of hidden layer in feed-forward network
    num_heads: int  # Number of heads to use in the Multi-Head Attention block
    num_layers: int  # Number of layers to use in the Transformer
    num_classes: int  # Number of classes to predict
    num_patches: int  # Maximum number of patches an image can have
    dropout_prob: float = 0.0  # Amount of dropout to apply in the feed-forward network

    def setup(self):
        # Layers/Networks
        self.input_layer = nn.Dense(self.embed_dim)
        self.transformer = [
            AttentionBlock(
                self.embed_dim, self.hidden_dim, self.num_heads, self.dropout_prob
            )
            for _ in range(self.num_layers)
        ]
        self.mlp_head = nn.Sequential([nn.LayerNorm(), nn.Dense(self.num_classes)])
        self.dropout = nn.Dropout(self.dropout_prob)

        # Parameters/Embeddings
        self.cls_token = self.param(
            "cls_token", nn.initializers.normal(stddev=1.0), (1, self.embed_dim)
        )
        self.pos_embedding = self.param(
            "pos_embedding",
            nn.initializers.normal(stddev=1.0),
            (1 + self.num_patches, self.embed_dim),
        )

        # self.norm_factor = self.param("norm_factor", nn.initializers.ones, (1, 1, 1))

    def __call__(self, x, train=True):
        T, _ = x.shape
        # x = x / self.norm_factor
        x = self.input_layer(x)

        # Add CLS token and positional encoding
        # cls_token = self.cls_token.repeat(B, axis=0)
        x = jnp.concatenate([self.cls_token, x], axis=0)
        x = x + self.pos_embedding[: T + 1]

        # Apply Transforrmer
        x = self.dropout(x, deterministic=not train)
        for attn_block in self.transformer:
            x = attn_block(x, train=train)

        # Perform classification prediction
        cls = x[0]
        out = self.mlp_head(cls)
        return out
