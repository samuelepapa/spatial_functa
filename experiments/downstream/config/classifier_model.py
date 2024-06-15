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
        config.embed_dim = 256
        config.hidden_dim = 512
        config.num_heads = 8
        config.num_layers = 6
        config.dropout_prob = 0.0
    elif model_name == "mlp":
        config.hidden_dim = 1024
        config.num_layers = 4
        config.dropout_prob = 0.0
    else:
        raise ValueError(f"Unknown model {model_name}")

    return config
