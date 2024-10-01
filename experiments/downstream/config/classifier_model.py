from ml_collections import config_dict


def get_config(model_name: str):
    config = config_dict.ConfigDict()

    # Model
    config.name = model_name

    if model_name == "resnet":
        config.num_blocks = [4, 4, 4]
        config.c_hidden = [128, 64, 32]
        config.act_fn_name = "relu"
    elif model_name == "transformer":
        config.embed_dim = 384
        config.hidden_dim = 768
        config.num_heads = 4
        config.num_layers = 12
        config.dropout_prob = 0.1
        config.num_patches = 4 * 4
    elif model_name == "mlp":
        config.hidden_dim = 1024
        config.num_layers = 3
        config.dropout_prob = 0.2
    else:
        raise ValueError(f"Unknown model {model_name}")

    return config
