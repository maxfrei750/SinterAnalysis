from pathlib import Path
from typing import Optional, Tuple

import fire

from paddle.custom_types import AnyPath
from paddle.deployment import run_model_on_dataset
from paddle.inspection import inspect_dataset
from paddle.lightning_modules import LightningMaskRCNN
from paddle.utilities import get_best_checkpoint_path, get_latest_log_folder_path


def test_model_on_dataset(
    config_name: str,
    subset: str,
    data_root: Optional[AnyPath] = "data",
    output_root: Optional[AnyPath] = "output",
    log_root: Optional[AnyPath] = "logs",
    model_id: Optional[str] = None,
    do_visualization: bool = True,
    gpus: Optional[int] = -1,
    initial_cropping_rectangle: Optional[Tuple[int, int, int, int]] = (0, 0, 1280, 960)
):
    """Performs a default analysis of a dataset using a given model. The
        analysis includes the filtering of border instances, a visualization
        and the measurement of the area equivalent diameter, as well as the
        minimum and maximum Feret diameters.

    Args:
        config_name (str): Name of the config utilized for the training.
        subset (str): Name of the subset to use.
        data_root (Optional[AnyPath], optional): Path of the data set folder,
            holding the subsets. Defaults to "data".
        output_root (Optional[AnyPath], optional): Root directory for output
            files. Defaults to "output".
        log_root (Optional[AnyPath], optional): Root directory for training
            log files.. Defaults to "logs".
        model_id (Optional[str], optional): ID of the model to use. If None,
            then the latest model will be used. Defaults to None.
        do_visualization (bool, optional): [description]. Defaults to True.
        gpus (int, optional): Specify, which compute device to use (see https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html#select-gpu-devices).
        initial_cropping_rectangle: If not None, [x_min, y_min, x_max, y_max] rectangle used for
        the cropping of images.
    """

    data_root = Path(data_root)
    log_root = Path(log_root) / config_name
    output_root = Path(output_root) / config_name
    model_checkpoint_path = get_checkpoint_path(log_root, model_id)

    output_root.mkdir(parents=True, exist_ok=True)

    model_checkpoint_path = Path(model_checkpoint_path)

    model = LightningMaskRCNN.load_from_checkpoint(str(model_checkpoint_path))

    run_model_on_dataset(
        model,
        output_root,
        data_root,
        subset,
        initial_cropping_rectangle=initial_cropping_rectangle,
        gpus=gpus,
    )

    if do_visualization:
        inspect_dataset(
            output_root,
            subset,
            do_display_box=False,
            do_display_label=False,
            do_display_score=True,
            score_threshold=0.5,
        )


def get_checkpoint_path(
    log_root: AnyPath, model_id: Optional[str] = None
) -> AnyPath:
    """Retrieve the best checkpoint of a model, based on the model id.

    :param log_root: Root directory for training log files.
    :param model_id: ID of the model to use. If None, then the latest model
        will be used.
    :return: Path to the best checkpoint of the model.
    """
    if model_id is None:
        model_id = get_latest_log_folder_path(log_root)
    model_root = Path(log_root) / model_id
    checkpoint_root = model_root / "checkpoints"
    checkpoint_path = get_best_checkpoint_path(checkpoint_root)
    return checkpoint_path


if __name__ == "__main__":
    fire.Fire(test_model_on_dataset)
