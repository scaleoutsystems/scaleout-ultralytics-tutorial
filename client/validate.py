import os
import sys
from ultralytics import YOLO
import tempfile

from fedn.utils.helpers.helpers import save_metrics

from model import load_parameters

def validate(in_model, out_json_path, data_yaml_path='data.yaml'):
    """Validate YOLO model.

    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_json_path: The path to save the output JSON to.
    :type out_json_path: str
    :param data_yaml_path: The path to the data file (YOLO dataset YAML file).
    :type data_yaml_path: str
    """
    # Load YOLOv8 model
    model = load_parameters(in_model)

    # Evaluate the model on both train and test datasets using YOLO's val() method
    with tempfile.TemporaryDirectory() as tmp_dir:
        train_results = model.val(data=data_yaml_path, split='train', verbose=False, exist_ok=True,project=tmp_dir)
        test_results = model.val(data=data_yaml_path, split='val', verbose=False, exist_ok=True,project=tmp_dir)

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


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python validate.py <in_model_path> <out_json_path> [data_yaml_path]")
        sys.exit(1)

    in_model_path = sys.argv[1]
    out_json_path = sys.argv[2]
    data_yaml_path = sys.argv[3] if len(sys.argv) > 3 else 'data.yaml'

    validate(in_model_path, out_json_path, data_yaml_path)