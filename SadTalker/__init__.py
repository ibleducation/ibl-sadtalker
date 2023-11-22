from time import strftime
from .inference import main

from argparse import Namespace

import logging
from pathlib import Path
from .download_models import download_models

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent


class Inference:
    """
    Wrapper class around inference cli module.
    """

    def __init__(
        self,
        driven_audio: Path | str,
        source_image: Path | str,
        ref_eyeblink: Path | str | None = None,
        ref_pose: Path | str | None = None,
        checkpoint_dir: Path | str = BASE_DIR / "checkpoints",
        result_dir: Path | str = "./result",
        output_filename: Path | str = None,
        device: str = "cpu",
        pose_style: int = 0,
        batch_size: int = 2,
        size: int = 256,
        expression_scale: float = 1.0,
        input_yaw: list[int] = None,
        input_pitch: list[int] = None,
        input_roll: list[int] = None,
        enhancer: str = None,
        background_enhancer: str = None,
        cpu: bool = False,
        face3dvis: bool = False,
        still: bool = False,
        preprocess: str = "crop",
        verbose: bool = False,
        old_version: bool = False,
        net_recon: str = "resnet50",
        ref_video: Path | str = None,
        use_last_fc: bool = False,
        bfm_folder: Path | str = BASE_DIR / "checkpoints" / "BFM_Fitting",
        bfm_model: str = "BFM_model_front.mat",
        focal: float = 1015.0,
        center: float = 112.0,
        camera_d: float = 10.0,
        z_near: float = 5.0,
        z_far: float = 15.0,
        init_path: str | None = None,
    ):
        """
        Initialize the class with the given parameters.

        Args:
            driven_audio (Path | str): The path to the driven audio file.
            source_image (Path | str): The path to the source image file.
            ref_eyeblink (Path | str | None, optional): The path to the reference eyeblink file. Defaults to None.
            ref_pose (Path | str | None, optional): The path to the reference pose file. Defaults to None.
            checkpoint_dir (Path | str, optional): The directory path for checkpoints. Defaults to BASE_DIR / "checkpoints".
            result_dir (Path | str, optional): The directory path for results. Defaults to "./result".
            output_filename (Path | str, optional): The output filename. Defaults to None.
            device (str, optional): The device to use. Defaults to "cpu".
            pose_style (int, optional): The pose style. Defaults to 0.
            batch_size (int, optional): The batch size. Defaults to 2.
            size (int, optional): The size. Defaults to 256.
            expression_scale (float, optional): The expression scale. Defaults to 1.0.
            input_yaw (list[int], optional): The input yaw. Defaults to None.
            input_pitch (list[int], optional): The input pitch. Defaults to None.
            input_roll (list[int], optional): The input roll. Defaults to None.
            enhancer (str, optional): The enhancer. Defaults to None.
            background_enhancer (str, optional): The background enhancer. Defaults to None.
            cpu (bool, optional): Whether to use CPU. Defaults to False.
            face3dvis (bool, optional): Whether to use face3d visualization. Defaults to False.
            still (bool, optional): Whether to generate still images. Defaults to False.
            preprocess (str, optional): The preprocess method. Defaults to "crop".
            verbose (bool, optional): Whether to output verbose information. Defaults to False.
            old_version (bool, optional): Whether to use the old version. Defaults to False.
            net_recon (str, optional): The recon network. Defaults to "resnet50".
            ref_video (Path | str, optional): The path to the reference video file. Defaults to None.
            use_last_fc (bool, optional): Whether to use the last fully connected layer. Defaults to False.
            bfm_folder (Path | str, optional): The BFM fitting folder. Defaults to BASE_DIR / "checkpoints" / "BFM_Fitting".
            bfm_model (str, optional): The BFM model. Defaults to "BFM_model_front.mat".
            focal (float, optional): The focal length. Defaults to 1015.0.
            center (float, optional): The center. Defaults to 112.0.
            camera_d (float, optional): The camera distance. Defaults to 10.0.
            z_near (float, optional): The near plane. Defaults to 5.0.
            z_far (float, optional): The far plane. Defaults to 15.0.
            init_path (str | None, optional): The initialization path. Defaults to None.

        Returns:
            None
        """
        if not output_filename:
            output_filename = strftime("%Y_%m_%d_%H.%M.%S") + ".mp4"
        checkpoint_dir = str(checkpoint_dir)
        self.args = Namespace(
            driven_audio=driven_audio,
            source_image=source_image,
            ref_eyeblink=ref_eyeblink,
            ref_pose=ref_pose,
            checkpoint_dir=checkpoint_dir,
            result_dir=result_dir,
            output_filename=output_filename,
            device=device,
            pose_style=pose_style,
            batch_size=batch_size,
            size=size,
            expression_scale=expression_scale,
            input_yaw=input_yaw,
            input_pitch=input_pitch,
            input_roll=input_roll,
            enhancer=enhancer,
            background_enhancer=background_enhancer,
            cpu=cpu,
            face3dvis=face3dvis,
            still=still,
            preprocess=preprocess,
            verbose=verbose,
            old_version=old_version,
            net_recon=net_recon,
            ref_video=ref_video,
            use_last_fc=use_last_fc,
            bfm_folder=bfm_model,
            bfm_model=bfm_model,
            focal=focal,
            center=center,
            camera_d=camera_d,
            z_near=z_near,
            z_far=z_far,
            init_path=init_path,
            current_root_path=str(BASE_DIR),
        )

        if not self.check_required_files(checkpoint_path=checkpoint_dir):
            raise Exception(
                "Required files not found. Please run and AI_PLUS_VIDEO_APP_AUTO_DOWNLOAD_SADTALKER_MODEL"
                "is set as False."
            )

    def check_required_files(self, checkpoint_path) -> bool:
        print("[checking for file download]")
        return download_models()

    def run(self) -> str:
        print("[EXECUTING INFERENCE.RUN]")

        try:
            main(self.args)
        except Exception as e:
            logger.exception(e)
        return str(Path(self.args.result_dir) / self.args.output_filename)
