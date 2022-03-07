from pathlib import Path

import fire

from custom_postprocessing import SimplifyGrainBoundaries
from paddle.custom_types import AnyPath
from paddle.data import MaskRCNNDataset
from paddle.postprocessing import Postprocessor
from paddle.postprocessing.postprocessingsteps import PickleAnnotation


def simplify_grain_boundaries(data_root: AnyPath, subset: str):
    data_root = Path(data_root)

    data_set = MaskRCNNDataset(
        data_root,
        subset=subset,
    )

    subset_folder_path = data_root / subset

    post_processing_steps = [
        SimplifyGrainBoundaries(),
        PickleAnnotation(subset_folder_path, exclude_keys=["masks"]),
    ]

    postprocessor = Postprocessor(
        data_set,
        post_processing_steps,
        progress_bar_description="Simplifying grain boundaries",
    )
    postprocessor.run()


if __name__ == "__main__":
    data_root = "output/ZnO"
    fire.Fire(simplify_grain_boundaries)
