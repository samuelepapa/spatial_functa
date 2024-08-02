import jax
import jax.numpy as jnp
from jax import Array


def interpolate_2d(feature_map: Array, coords: Array, interp_mode="1-NN") -> Array:
    # @jax.jit
    def nn_interp(feature_map: Array, coords: Array):
        height, width, channels = feature_map.shape
        # get the x and y coordinates
        x, y = jnp.split(coords, 2, axis=-1)
        x = x.reshape(-1)
        y = y.reshape(-1)
        # x and y are between -1 and 1, so we need to convert them to the pixel coordinates
        x = x * width
        y = y * height
        # find the closes pixel based on coords by rounding
        x = jnp.floor(x).astype(jnp.int32)
        y = jnp.floor(y).astype(jnp.int32)

        # return the pixel value at the locations
        return feature_map[y, x]

    # @jax.jit
    def linear_interp(feature_map: Array, coords: Array):
        # feature_map has dimension [height, width, channels]
        # coords has dimension [num_points, 2] and is floating point coordinates
        # returns [num_points, channels]
        # perform the bilinear interpolation
        # get the height and width of the feature map
        # pad 1 pixel around the feature map
        # feature_map = jnp.pad(feature_map, ((2, 2), (2, 2), (0, 0)), mode="reflect")
        height, width, _ = feature_map.shape
        # get the x and y coordinates
        x, y = jnp.split(coords, 2, axis=-1)
        # flatten them
        x = x.reshape(-1)
        y = y.reshape(-1)
        # # convert the x and y coordinates to pixel coordinates
        x = x * (width - 1)
        y = y * (height - 1)
        # get the floor and ceil of the x and y coordinates
        x0 = jnp.floor(x).astype(jnp.int32)
        x1 = jnp.minimum(x0 + 1, width + 1)
        y0 = jnp.floor(y).astype(jnp.int32)
        y1 = jnp.minimum(y0 + 1, height + 1)

        # get the pixel values at the corners
        Ia = feature_map[y0, x0]
        Ib = feature_map[y1, x0]
        Ic = feature_map[y0, x1]
        Id = feature_map[y1, x1]
        # calculate the weights
        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)
        # calculate the interpolated pixel values
        interpolated = (
            wa[:, None] * Ia + wb[:, None] * Ib + wc[:, None] * Ic + wd[:, None] * Id
        )
        return interpolated

    if interp_mode == "1-NN":
        return nn_interp(feature_map, coords)
    elif interp_mode == "linear":
        return linear_interp(feature_map, coords)


if __name__ == "__main__":
    import math

    import matplotlib.pyplot as plt
    from pathlib import Path

    vis_folder = Path("visualizations")

    # a test for the interpolation
    # create a feature map
    rng = jax.random.PRNGKey(0)
    latent_spatial_dim = 8
    feature_map = jax.random.uniform(rng, (latent_spatial_dim, latent_spatial_dim, 3))
    interpolation_type = "linear"
    width = 32
    height = 32

    # coordinates
    center_pixel_x = 0.5 / width
    center_pixel_y = 0.5 / height
    x = jnp.linspace(center_pixel_y, 1 - center_pixel_y, height)
    y = jnp.linspace(center_pixel_x, 1 - center_pixel_x, width)
    x, y = jnp.meshgrid(x, y)
    coords = jnp.stack([x, y], axis=-1)
    coords = coords.reshape(-1, 2)

    # interpolate the feature map
    interpolated = interpolate_2d(feature_map, coords, interp_mode=interpolation_type)

    x_coord, y_coord = jnp.split(coords, 2, axis=-1)
    if interpolation_type == "1-NN":
        x_coord = (latent_spatial_dim * x_coord) % 1.0
        y_coord = (latent_spatial_dim * y_coord) % 1.0
        x = jnp.concat([x_coord, y_coord], axis=-1)
    else:
        num_bits = int(jnp.ceil(jnp.log2(width)))
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
        x_coord_binary = jnp.floor(x_coord * width).astype(jnp.uint16)
        y_coord_binary = jnp.floor(y_coord * height).astype(jnp.uint16)
        x_coord_binary = to_binary(x_coord_binary, num_bits, axis=1).reshape(-1, num_bits)
        y_coord_binary = to_binary(y_coord_binary, num_bits, axis=1).reshape(-1, num_bits)
        coords_binary = jnp.concatenate([x_coord_binary, y_coord_binary], axis=-1)
        x = coords_binary


    # plot the interpolated image
    plt.figure()
    plt.imshow(interpolated.reshape(height, width, 3))
    plt.savefig(vis_folder/"interpolated.png")

    plt.figure(figsize=(5, 5))

    plt.scatter(coords[:, 0], -coords[:, 1], c=interpolated, s=30)
    plt.savefig(vis_folder/"scatter_interp.png")

    plt.figure()
    plt.imshow(feature_map)
    plt.savefig(vis_folder/"feature_map.png")
