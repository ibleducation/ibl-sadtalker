import subprocess
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

current_directory = Path(__file__).parent


def check_checkpoint_files():
    files = [
        "checkpoints/mapping_00109-model.pth.tar",
        "checkpoints/mapping_00229-model.pth.tar",
        "checkpoints/SadTalker_V0.0.2_256.safetensors",
        "checkpoints/SadTalker_V0.0.2_512.safetensors",
        "gfpgan/weights/alignment_WFLW_4HG.pth",
        "gfpgan/weights/detection_Resnet50_Final.pth",
        "gfpgan/weights/GFPGANv1.4.pth",
        "gfpgan/weights/parsing_parsenet.pth",
    ]
    for file in files:
        if not (current_directory / file).is_file():
            return False
    return True

def download_models(directory: Path | str = current_directory,) -> bool:
    """
    Downloads models from a specified directory.
    This calls the bash script packaged as part of SadTalker.
    An exception is raised if the download fails.
    Args:
        directory (Path|str): The directory to download the models to. Defaults to the current directory.

    Returns:
        bool: True if the download was successful, False otherwise.
    """
    if not check_checkpoint_files():
        print("Calling downloads. with path: " + str(directory))
        raise Exception("Calling downloads " + str(directory))
        # subprocess.call(
        #     ["sh", "./scripts/download_models.sh"], cwd=str(directory)
        # )
    return check_checkpoint_files()
