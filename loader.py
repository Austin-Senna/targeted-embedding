import os
import glob
import logging
import yaml
import sys


def load(path):
    path = os.path.join("/projects/bgbh/datasets", path)
    list_files = glob.glob(f'{path}/**/*.wav', recursive=True)
    list_files.sort()

    if not list_files:
        print(f"Warning: No files found in {path}")

    return list_files

def setup_logging(run_type):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(f"{run_type}_extraction_run.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_save_path(base_folder, file_path, audio_folder, folder_mode, eval_type, run_type):
    """Determines where the .npy file should be saved based on the mode."""
    output_folder = os.path.join(f"{base_folder}_{eval_type}", run_type)
    file_name = os.path.basename(file_path)
    save_name = file_name.replace('.wav', '.npy') 
    
    if folder_mode == "CREATE":
        # Reconstruct the absolute path that load() used
        absolute_audio_folder = os.path.join("/projects/bgbh/datasets", audio_folder)
        rel_path = os.path.relpath(file_path, absolute_audio_folder)
        rel_dir = os.path.dirname(rel_path)
        save_folder = os.path.join(output_folder, rel_dir)
    else:
        save_folder = output_folder
        
    os.makedirs(save_folder, exist_ok=True)
    return os.path.join(save_folder, save_name)

