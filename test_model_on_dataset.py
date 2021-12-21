from pathlib import Path
from typing import Optional

import fire

from paddle.custom_types import AnyPath
from paddle.deployment import run_model_on_dataset
from paddle.lightning_modules import LightningMaskRCNN
from paddle.utilities import (
    get_best_checkpoint_path,
    get_latest_log_folder_path,
)


def test_model_on_dataset(
    config_name: str,
    subset: str,
    data_root: Optional[AnyPath] = "data",
    output_root: Optional[AnyPath] = "output",
    log_root: Optional[AnyPath] = "logs",
    model_id: Optional[str] = None,
):
    """Performs a default analysis of a dataset using a given model. The
        analysis includes the filtering of border instances, a visualization
        and the measurement of the area equivalent diameter, as well as the
        minimum and maximum Feret diameters.

    :param config_name: Name of the config utilized for the training.
    :param subset: Name of the subset to use.
    :param data_root: Path of the data set folder, holding the subsets.
    :param output_root: Root directory for output files.
    :param log_root: Root directory for training log files.
    :param model_id: ID of the model to use. If None, then the latest model
        will be used.
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
