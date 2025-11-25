from scaleoututil.helpers.helpers import get_helper
from ultralytics import YOLO
import torch
import collections
import tempfile
import io

from scaleout import ScaleoutModel

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


def load_parameters(model: ScaleoutModel):
    """Load model parameters from file and populate model.

    param model_path: The path to load from.
    :type model_path: str
    :return: The loaded model.
    :rtype: torch.nn.Module
    """
    
    weights = model.get_model_params(helper)
    yolo_model = compile_model()
    torch_model = yolo_model.model.model

    keys = list(torch_model.state_dict().keys())

    if len(weights) != len(keys):
        raise ValueError(f"Mismatch in number of parameters: expected {len(keys)}, got {len(weights)}")

    state_dict = collections.OrderedDict(
        {key: torch.tensor(x) for key, x in zip(keys, weights)}
    )

    torch_model.load_state_dict(state_dict, strict=False)
    yolo_model.ckpt = {'model': torch_model}
    
    return yolo_model

def save_parameters(model):
    """Save model parameters to file.

    :param model: The model to serialize.
    :type model: torch.nn.Module
    :param buffer: The buffer to write to.
    :type buffer: io.BytesIO
    :param out_path: The path to save to.
    :type out_path: str
    """
    weights = [val.cpu().numpy() for _, val in model.state_dict().items()]
    return ScaleoutModel.from_model_params(weights, helper)

def init_seed(out_path):
    model = compile_model()
    weights = [val.cpu().numpy() for _, val in model.state_dict().items()]
    helper.save(weights, out_path)

def build():
    init_seed('seed.npz')

if __name__ == "__main__":
    init_seed('../seed.npz')