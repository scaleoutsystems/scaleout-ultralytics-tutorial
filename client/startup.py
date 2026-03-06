from scaleout import EdgeClient, ScaleoutModel
from scaleoututil.helpers.helpers import get_helper

from model import load_parameters, save_parameters, get_best_device
from data import get_dataset_size

import os
import yaml
import tempfile
import io

HELPER_MODULE = "numpyhelper"
helper = get_helper(HELPER_MODULE)

def startup(client: EdgeClient):
    MyClient(client)

class MyClient(EdgeClient):
    def __init__(self, client: EdgeClient):
        self.client = client
        client.set_train_callback(self.train)
        client.set_validate_callback(self.validate)

    def train(self, model: ScaleoutModel, settings, epochs=10, data_yaml_path='client_config.yaml', batch_size=16):
        """Complete a model update using YOLOv8.

        Load model parameters from in_model_path (managed by the Scaleout client),
        perform a model update, and write updated parameters
        to out_model_path (picked up by the Scaleout server).

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

        # Get the best device for training
        device = get_best_device()

        # Load YOLOv8 model
        model = load_parameters(model)

        # Load the client configuration 
        config_path = os.path.join(os.path.dirname(__file__), 'client_config.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            epochs = config.get('local_epochs', epochs)
            batch_size = config.get('batch_size', batch_size)
        else:
            print(f"Client config file not found at {config_path}. Using default epochs ({epochs}) and batch size ({batch_size}).")

        model.add_callback('on_train_batch_start', lambda trainer: self.client.check_task_abort())

        # Train the model and remove the unnecessary files
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.train(data=data_yaml_path, device=device, epochs=epochs, batch=batch_size, optimizer='SGD', warmup_epochs=0, lrf=1.0, lr0=0.005, verbose=False, exist_ok=True, save=False, plots=False, val=False, project=tmp_dir)

        # Save the updated model to the output path
        out_model = save_parameters(model)

        # Metadata needed for aggregation server side
        metadata = {
            "num_examples": get_dataset_size(data_yaml_path, 'train'),  # Get number of examples
        }

        return out_model, {"training_metadata": metadata}

    def validate(self, model, data_yaml_path='client_config.yaml'):
        """Validate YOLO model.

        :param in_model_path: The path to the input model.
        :type in_model_path: str
        :param out_json_path: The path to save the output JSON to.
        :type out_json_path: str
        :param data_yaml_path: The path to the data file (YOLO dataset YAML file).
        :type data_yaml_path: str
        """
        # Load YOLOv8 model
        model = load_parameters(model)

        # Evaluate the model on both train and test datasets using YOLO's val() method
        with tempfile.TemporaryDirectory() as tmp_dir:
            train_results = model.val(data=data_yaml_path, split='train', verbose=False, exist_ok=True, plots=False, save=False, project=tmp_dir)
            test_results = model.val(data=data_yaml_path, split='val', verbose=False, exist_ok=True, plots=False, save=False, project=tmp_dir)

        # Extract metrics from the results
        report = {
            "training_recall": train_results.results_dict['metrics/recall(B)'],
            "training_precision": train_results.results_dict['metrics/precision(B)'],
            "training_mAP50": train_results.results_dict['metrics/mAP50(B)'],  # mAP for training data
            "training_mAP50-95": train_results.results_dict['metrics/mAP50-95(B)'],  # mAP for training data
            "test_recall": test_results.results_dict['metrics/recall(B)'],
            "test_precision": test_results.results_dict['metrics/precision(B)'],
            "test_mAP50": test_results.results_dict['metrics/mAP50(B)'],  # mAP for testing data
            "test_mAP50-95": test_results.results_dict['metrics/mAP50-95(B)'],  # mAP for testing data
        }

        return report