import os
import yaml
from ultralytics import settings
from pathlib import Path

def get_yolo_datasets_path():
    """ Get the path to the YOLO datasets directory.

    :return: The path to the YOLO datasets directory.
    :rtype: str
    """    
    return settings['datasets_dir']

def get_dataset_size(yaml_path='client_config.yaml', split='train'):
    """ Counts images in a YOLO dataset split using fast directory scanning.
    
    Args:
        yaml_path (str): Path to the data.yaml file.
        split (str): One of 'train', 'val', or 'test'.
    
    Returns:
        int: Number of image files.
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
        
    rel_path = data.get(split)
    if not rel_path:
        return 0 

    yaml_dir = Path(yaml_path).parent
    dataset_dir = (yaml_dir / rel_path).resolve()
    
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Directory not found: {dataset_dir}")

    count = 0
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    try:
        with os.scandir(dataset_dir) as entries:
            for entry in entries:
                if entry.is_file() and Path(entry.name).suffix.lower() in valid_exts:
                    count += 1
    except OSError as e:
        print(f"Error reading directory {dataset_dir}: {e}")
        
    return count