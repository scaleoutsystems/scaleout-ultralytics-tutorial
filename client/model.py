from fedn.utils.helpers.helpers import get_helper
from ultralytics import YOLO
import torch
import collections
import tempfile
import io

HELPER_MODULE = "numpyhelper"
helper = get_helper(HELPER_MODULE)

def get_best_device():
    """
    Get the best device for training.
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

def compile_model():
    """Compile the YOLO model.
    """
    device = get_best_device()
    return YOLO('model.yaml').to(device)


def load_parameters(model_path):
    """Load model parameters from file and populate model.

    param model_path: The path to load from.
    :type model_path: str
    :return: The loaded model.
    :rtype: torch.nn.Module
    """
    
    parameters_np = helper.load(model_path)
    model = compile_model()
    params_dict = zip(model.state_dict().keys(), parameters_np)
    state_dict = collections.OrderedDict({key: torch.tensor(x) for key, x in params_dict})
    model.load_state_dict(state_dict, strict=False)
    with tempfile.NamedTemporaryFile(suffix='.pt') as tmp_file:
        torch.save(model,tmp_file.name)
        model = YOLO(tmp_file.name)
    return model

def save_parameters(model, out_path=None):
    """Save model parameters to file.

    :param model: The model to serialize.
    :type model: torch.nn.Module
    :param buffer: The buffer to write to.
    :type buffer: io.BytesIO
    :param out_path: The path to save to.
    :type out_path: str
    """
    parameters_np = [val.cpu().numpy() for _, val in model.state_dict().items()]
    params = helper.save(parameters_np, out_path)
    return params

def init_seed(out_path):
    model = compile_model()
    save_parameters(model, io.BytesIO(), out_path)

if __name__ == "__main__":
    init_seed('../seed.npz')