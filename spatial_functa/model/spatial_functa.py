import math
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    Literal,
)

import jax
from flax import linen as nn
from jax import numpy as jnp
from jax.nn.initializers import Initializer


import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
from jax import core, dtypes, random

from spatial_functa.model.interpolation import interpolate_2d
from spatial_functa.opt_builder import build_lr_scheduler

Array = jnp.ndarray
KeyArray = Array

Shape = Sequence[int | Any]

class TrainState(train_state.TrainState):
    # Adding rng key for masking
    rng: Any = None

from jax._src.nn.initializers import _compute_fans
# def _compute_fans(
#     shape: Sequence[int],
#     in_axis: Union[int, Sequence[int]] = -2,
#     out_axis: Union[int, Sequence[int]] = -1,
#     batch_axis: Union[int, Sequence[int]] = (),
# ):
#     """Compute effective input and output sizes for a linear or convolutional layer.

#     Axes not in in_axis, out_axis, or batch_axis are assumed to constitute the "receptive field" of
#     a convolution (kernel spatial dimensions).
#     """
#     if shape.rank <= 1:
#         raise ValueError(
#             f"Can't compute input and output sizes of a {shape.rank}"
#             "-dimensional weights tensor. Must be at least 2D."
#         )

#     if isinstance(in_axis, int):
#         in_size = shape[in_axis]
#     else:
#         in_size = math.prod([shape[i] for i in in_axis])
#     if isinstance(out_axis, int):
#         out_size = shape[out_axis]
#     else:
#         out_size = math.prod([shape[i] for i in out_axis])
#     if isinstance(batch_axis, int):
#         batch_size = shape[batch_axis]
#     else:
#         batch_size = math.prod([shape[i] for i in batch_axis])
#     receptive_field_size = shape.total / in_size / out_size / batch_size
#     fan_in = in_size * receptive_field_size
#     fan_out = out_size * receptive_field_size
#     return fan_in, fan_out


def custom_uniform(
    numerator: Any = 6,
    mode="fan_in",
    dtype: Any = jnp.float_,
    in_axis: Union[int, Sequence[int]] = -2,
    out_axis: Union[int, Sequence[int]] = -1,
    batch_axis: Sequence[int] = (),
    distribution="uniform",
) -> Initializer:
    """Builds an initializer that returns real uniformly-distributed random arrays.

    Args:
      scale: the upper and lower bound of the random distribution.
      dtype: optional; the initializer's default dtype.

    Returns:
      An initializer that returns arrays whose values are uniformly distributed in
      the range ``[-range, range)``.
    """

    def init(key: KeyArray, shape: Shape, dtype: Any = dtype) -> Any:
        dtype = dtypes.canonicalize_dtype(dtype)
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis, batch_axis)
        if mode == "fan_in":
            denominator = fan_in
        elif mode == "fan_out":
            denominator = fan_out
        elif mode == "fan_avg":
            denominator = (fan_in + fan_out) / 2
        else:
            raise ValueError(f"invalid mode for variance scaling initializer: {mode}")
        if distribution == "uniform":
            return random.uniform(
                key,
                shape,
                dtype,
                minval=-jnp.sqrt(numerator / denominator),
                maxval=jnp.sqrt(numerator / denominator),
            )
        elif distribution == "normal":
            return random.normal(key, shape, dtype) * jnp.sqrt(numerator / denominator)
        elif distribution == "uniform_squared":
            return random.uniform(
                key,
                shape,
                dtype,
                minval=-numerator / denominator,
                maxval=numerator / denominator,
            )
        else:
            raise ValueError(
                f"invalid distribution for variance scaling initializer: {distribution}"
            )

    return init


class MetaSGDLr(nn.Module):
    shape: Tuple[int, int, int]
    lr_init_range: Tuple[float, float] = (0.0001, 0.001)
    lr_clip_range: Tuple[float, float] = (0.0, 1.0)
    lr_scaling: float = 256.0

    @nn.compact
    def __call__(self) -> Array:
        lrs = self.param(
            "lrs",
            lambda key, shape: jax.random.uniform(
                key,
                shape,
                jnp.float32,
                self.lr_init_range[0],
                self.lr_init_range[1],
            ),
            self.shape,
        )
        return jax.tree_util.tree_map(lambda x: self.lr_scaling*jnp.clip(x, *self.lr_clip_range), lrs)


class LatentVector(nn.Module):
    """
    Latent vector used to condition the neural field.

    This latent vector is also used to process downstream tasks.
    """

    latent_dim: int
    latent_spatial_dim: int

    @nn.compact
    def __call__(
        self,
    ) -> Array:
        latent_vector = self.param(
            "latent_vector",
            nn.initializers.zeros,
            (
                self.latent_spatial_dim,
                self.latent_spatial_dim,
                self.latent_dim,
            ),
        )

        return latent_vector


class MLP(nn.Module):
    """Simple MLP with custom activation function."""

    layer_sizes: Sequence[int]
    activation: Callable[[Array], Array] = nn.relu

    @nn.compact
    def __call__(self, x: Array) -> Array:
        for i, size in enumerate(self.layer_sizes[:-1]):
            x = nn.Dense(
                size,
                kernel_init=jax.nn.initializers.truncated_normal(1/jnp.sqrt(x.shape[-1])),
                bias_init=jax.nn.initializers.zeros,
                name=f"mlp_linear_{i}",
            )(x)
            x = self.activation(x)
        return nn.Dense(
            self.layer_sizes[-1],
            kernel_init=jax.nn.initializers.truncated_normal(1/jnp.sqrt(x.shape[-1])),
            bias_init=jax.nn.initializers.zeros,
            name=f"mlp_linear_{len(self.layer_sizes) - 1}",
        )(x)


class LatentToModulation(nn.Module):
    layer_sizes: Sequence[int]
    num_modulation_layers: int
    modulation_dim: int
    input_dim: int
    shift_modulate: bool = True
    scale_modulate: bool = True
    activation: Callable[[Array], Array] = nn.relu

    def setup(self):
        if self.shift_modulate and self.scale_modulate:
            self.modulations_per_unit = 2
        elif self.shift_modulate or self.scale_modulate:
            self.modulations_per_unit = 1
        else:
            raise ValueError(
                "At least one of shift_modulate or scale_modulate must be True"
            )

        self.modulations_per_layer = self.modulations_per_unit * self.modulation_dim
        self.modulation_output_size = (
            self.modulations_per_layer * self.num_modulation_layers
        )

        # create a MLP to process the latent vector based on self.layer_sizes and self.modulation_output_size
        self.mlp = MLP(
            layer_sizes=self.layer_sizes + (self.modulation_output_size,),
            activation=self.activation,
        )

    def __call__(self, x: Array) -> Dict[str, Array]:
        x = self.mlp(x)
        # Split the output into scale and shift modulations
        if self.modulations_per_unit == 2:
            scale, shift = jnp.split(x, 2, axis=-1)
            scale = (
                scale.reshape(
                    (self.num_modulation_layers, self.input_dim, self.modulation_dim)
                )
                + 1
            )
            shift = shift.reshape(
                (self.num_modulation_layers, self.input_dim, self.modulation_dim)
            )
            return {"scale": scale, "shift": shift}
        else:
            x = x.reshape(
                (self.num_modulation_layers, self.input_dim, self.modulation_dim)
            )
            if self.shift_modulate:
                return {"shift": x}
            elif self.scale_modulate:
                return {"scale": x + 1}


class SirenLayer(nn.Module):
    output_dim: int
    omega_0: float
    is_first_layer: bool = False
    is_last_layer: bool = False
    apply_activation: bool = True

    def setup(self):
        c = 1 if self.is_first_layer else 6 / self.omega_0**2
        distrib = "uniform_squared" if self.is_first_layer else "uniform"
        self.linear = nn.Dense(
            features=self.output_dim,
            use_bias=True,
            kernel_init=custom_uniform(
                numerator=c, mode="fan_in", distribution=distrib
            ),
            bias_init=nn.initializers.zeros,
        )

    def __call__(self, x):
        after_linear = self.linear(x)
        if self.apply_activation:
            return jnp.sin(self.omega_0 * after_linear)
        else:
            return after_linear


class SIREN(nn.Module):
    input_dim: int
    output_dim: int
    hidden_dim: int
    latent_dim: int
    latent_spatial_dim: int
    num_layers: int
    omega_0: float
    modulation_hidden_dim: int
    modulation_num_layers: int
    image_width: int
    image_height: int
    learn_lrs: bool = False
    lr_init_range: Tuple[float, float] = (0.0001, 0.001)
    lr_clip_range: Tuple[float, float] = (0.0, 1.0)
    lr_shape_type: Literal['full', 'spatial', 'constant'] = 'spatial'
    lr_scaling: float = 256.0
    shift_modulate: bool = True
    scale_modulate: bool = True
    interpolation_type: str = "linear"

    def setup(self):
        self.conv_blocks = [
            nn.Conv(
                self.modulation_hidden_dim, 
                (3, 3),
                (1, 1),
                padding="SAME",
                kernel_init=nn.initializers.variance_scaling(1.0, "fan_in", "truncated_normal"),
            )
                # kernel_init=nn.initializers.truncated_normal(1/jnp.sqrt(self.latent_spatial_dim*self.latent_spatial_dim*self.latent_dim)))
        ]
        self.latent_to_modulation = LatentToModulation(
            input_dim=self.input_dim if self.latent_spatial_dim > 1 else 1,
            layer_sizes=[self.hidden_dim] * self.modulation_num_layers,
            num_modulation_layers=self.num_layers - 1,
            modulation_dim=self.hidden_dim,
            scale_modulate=self.scale_modulate,
            shift_modulate=self.shift_modulate,
        )

        if self.learn_lrs:
            if self.lr_shape_type == 'full':
                shape = (
                    self.latent_spatial_dim,
                    self.latent_spatial_dim,
                    self.latent_dim,
                )
            elif self.lr_shape_type == 'spatial':
                shape = (
                    self.latent_spatial_dim,
                    self.latent_spatial_dim,
                    1,
                )
            elif self.lr_shape_type == 'constant':
                shape = (1,1,1)
            else:
                raise ValueError(f"Invalid lr_shape_type: {self.lr_shape_type}, only valid options are full, spatial and constant")

            self.lrs = MetaSGDLr(
                shape=shape,
                lr_init_range=self.lr_init_range,
                lr_clip_range=self.lr_clip_range,
                lr_scaling=self.lr_scaling,
            )

        self.kernel_net = (
            [
                SirenLayer(
                    output_dim=self.hidden_dim,
                    omega_0=self.omega_0,
                    is_first_layer=True,
                    apply_activation=False,
                )
            ]
            + [
                SirenLayer(
                    output_dim=self.hidden_dim,
                    omega_0=self.omega_0,
                    is_first_layer=False,
                    apply_activation=False,
                )
                for _ in range(self.num_layers - 2)
            ]
            + [
                nn.Dense(
                    features=self.output_dim,
                    use_bias=True,
                    kernel_init=custom_uniform(
                        numerator=6 / self.omega_0**2,
                        mode="fan_in",
                        distribution="uniform",
                    ),
                    bias_init=nn.initializers.zeros,
                )
            ]
        )

    def __call__(self, x, latent_feat_map):
        if self.learn_lrs:
            lrs = self.lrs()

        if latent_feat_map.shape[0] > 1:
            for layer_num, layer in enumerate(self.conv_blocks):
                latent_feat_map = layer(latent_feat_map)

            latent_vector = interpolate_2d(latent_feat_map, x, self.interpolation_type)

            if self.interpolation_type == "1-NN":
                x_coord, y_coord = jnp.split(x, 2, axis=-1)
                x_coord = (self.latent_spatial_dim * x_coord) % 1.0
                y_coord = (self.latent_spatial_dim * y_coord) % 1.0
                x = jnp.concat([x_coord, y_coord], axis=-1)
            elif self.interpolation_type == "linear":
                # calculate number of bits necessary to encode width and height
                x_coord, y_coord = jnp.split(x, 2, axis=-1)
                resolution = max(self.image_width, self.image_height)
                num_bits = int(math.ceil(math.log2(resolution)))
                def to_binary(arr, num_bits, axis=None, count=None):
                    bits = jnp.asarray(1) << jnp.arange(num_bits, dtype='uint8')
                    if axis is None:
                        arr = jnp.ravel(arr)
                        axis = 0
                    arr = jnp.swapaxes(arr, axis, -1)
                    unpacked = ((arr[..., None] & jnp.expand_dims(bits, tuple(range(arr.ndim)))) > 0).astype('uint8')
                    unpacked = unpacked.reshape(unpacked.shape[:-2] + (-1,))
                    if count is not None:
                        if count > unpacked.shape[-1]:
                            unpacked = jnp.pad(unpacked, [(0, 0)] * (unpacked.ndim - 1) + [(0, count - unpacked.shape[-1])])
                        else:
                            unpacked = unpacked[..., :count]
                    return jnp.swapaxes(unpacked, axis, -1)
                x_coord_binary = jnp.floor(x_coord * self.image_width).astype(jnp.uint16)
                y_coord_binary = jnp.floor(y_coord * self.image_height).astype(jnp.uint16)
                x_coord_binary = to_binary(x_coord_binary, num_bits, axis=1).reshape(-1, num_bits)
                y_coord_binary = to_binary(y_coord_binary, num_bits, axis=1).reshape(-1, num_bits)
                coords_binary = jnp.concatenate([x_coord_binary, y_coord_binary], axis=-1)
                x = coords_binary

        else:
            latent_vector = latent_feat_map  # jnp.broadcast_to(latent_feat_map[0,0], x.shape[:-1] + (self.latent_dim,))

        modulations = self.latent_to_modulation(latent_vector)

        for layer_num, layer in enumerate(self.kernel_net):
            x = layer(x)
            if layer_num < self.num_layers - 1:
                x = self.modulate(x, modulations, layer_num)
                x = jnp.sin(self.omega_0 * x)

        return x + 0.5

    def modulate(
        self, x: Array, modulations: Dict[str, Array], layer_num: int
    ) -> Array:
        """Modulates input according to modulations.

        Args:
            x: Hidden features of MLP.
            modulations: Dict with keys 'scale' and 'shift' (or only one of them)
            containing modulations.

        Returns:
            Modulated vector.
        """
        if "scale" in modulations:
            x = modulations["scale"][layer_num] * x
        if "shift" in modulations:
            x = x + modulations["shift"][layer_num]
        return x
