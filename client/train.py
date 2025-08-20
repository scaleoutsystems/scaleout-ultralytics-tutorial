import os
import sys
from ultralytics import YOLO
import torch
from fedn.utils.helpers.helpers import save_metadata
import tempfile
import io

from model import load_parameters, save_parameters
from data import get_train_size
import yaml

def train(in_model, settings, epochs=10, data_yaml_path='data.yaml', batch_size=16):
    """Complete a model update using YOLOv8.

    Load model parameters from in_model_path (managed by the FEDn client),
    perform a model update, and write updated parameters
    to out_model_path (picked up by the FEDn server).

    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_model_path: The path to save the output model to.
    :type out_model_path: str
    :param data_path: The path to the data file (YOLO dataset YAML file).
    :type data_path: str
    :param batch_size: The batch size to use.
    :type batch_size: int
    :param epochs: The number of epochs to train.
    :type epochs: int
    :param lr: The learning rate to use.
    :type lr: float
    """

    # Load YOLOv8 model
    model = load_parameters(in_model)

    # Load the client configuration 
    config_path = os.path.join(os.path.dirname(__file__), 'client_config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        epochs = config.get('local_epochs', epochs)
        batch_size = config.get('batch_size', batch_size)
    else:
        print(f"Client config file not found at {config_path}. Using default epochs ({epochs}) and batch size ({batch_size}).")

    # Train the model and remove the unnecessary files
    with tempfile.TemporaryDirectory() as tmp_dir:
        model.train(data=data_yaml_path, epochs=epochs,batch=batch_size,verbose=False,exist_ok=True, project=tmp_dir)

    # Save the updated model to the output path
    out_model = save_parameters(model, io.BytesIO())

    # Metadata needed for aggregation server side
    metadata = {
        "num_examples": get_train_size(data_yaml_path),  # Get number of examples
    }

    return out_model, {"training_metadata": metadata}

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python train.py <in_model_path> <out_model_path> [epochs]")
        sys.exit(1)

    in_model_path = sys.argv[1]
    out_model_path = sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    train(in_model_path, out_model_path, epochs)