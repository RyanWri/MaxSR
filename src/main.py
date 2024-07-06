from hp.hyper_params import load_config
from layers.max_sr import MaxSR


def create_model(config):
    model_config = config["model"]
    sfeb_config = config["sfeb"]
    ada_grid_att_config = config["adaptive_grid_attention"]
    hffb_config = config["hffb"]
    rb_config = config["rb"]

    model = MaxSR(
        in_channels=model_config["in_channels"],
        out_channels=model_config["out_channels"],
        num_blocks=model_config["num_blocks"],
        grid_size=ada_grid_att_config["grid_size"],
        hidden_dim=model_config["hidden_dim"],
        dropout=model_config["dropout"],
    )

    return model


if __name__ == "__main__":
    # Load config and create model
    config = load_config("config.yaml")
    model = create_model(config)
