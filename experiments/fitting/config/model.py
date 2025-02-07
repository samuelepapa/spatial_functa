from ml_collections import config_dict


def get_config(model_name: str):
    config = config_dict.ConfigDict()

    # Model
    config.name = model_name

    if model_name == "latent":
        config.num_layers = 15
        config.hidden_dim = 512
        config.omega_0 = 30
        config.modulation_num_layers = 0
        config.modulation_hidden_dim = 1024
        config.latent_spatial_dim = 1
        config.latent_dim = 512
        config.learn_lrs = True
        config.scale_modulate = False
        config.shift_modulate = True
        config.interpolation_type = "1-NN"  # "linear" or "1-NN"
    else:
        raise ValueError(f"Unknown model {model_name}")

    return config
