from pathlib import Path

def get_config():
    return {
        # "filter_size": 64, #
        "output_size": 512,
        "kernel_size": 12,
        # "kernel_size_2": 3, #
        "batch_size": 32,
        "epochs": 1000,
        "lr": 10**-5,
        "num_layers": 12,
        "num_heads": 8,
        "num_classes": 6,
        "dropout": 0.1,
        "model_folder": "weights",
        "model_basename": "convtmodel_",
        "preload": "latest",
        "experiment_name": "runs/convtmodel"
    }


def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

def latest_weights_file_path(config):
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(weights_files[0])