from pathlib import Path

import fire

from custom_postprocessing import SimplifyGrainBoundaries
from paddle.custom_types import AnyPath
from paddle.data import MaskRCNNDataset
from paddle.postprocessing import (
    MeasureMaskProperties,
    PickleAnnotation,
    Postprocessor,
    calculate_areas,
    calculate_distances_to_image_border,
    calculate_major_axis_lengths,
    calculate_maximum_feret_diameters,
    calculate_minimum_feret_diameters,
    calculate_minor_axis_lengths,
    calculate_perimeters,
)


def measure_grain_properties(data_root: AnyPath, subset: str) -> None:
    data_root = Path(data_root)

    data_set = MaskRCNNDataset(
        data_root,
        subset=subset,
    )

    subset_folder_path = data_root / subset

    measurement_fcns = {
        "mask_feret_diameter_max": calculate_maximum_feret_diameters,
        "mask_feret_diameter_min": calculate_minimum_feret_diameters,
        "mask_area": calculate_areas,
        "mask_perimeter": calculate_perimeters,
        "mask_axis_length_min": calculate_minor_axis_lengths,
        "mask_axis_length_max": calculate_major_axis_lengths,
        "mask_distance_to_image_border": calculate_distances_to_image_border,
    }

    post_processing_steps = [
        MeasureMaskProperties(measurement_fcns=measurement_fcns),
        SimplifyGrainBoundaries(),
        PickleAnnotation(
            subset_folder_path,
            exclude_keys=[
                "masks",
                "area",
                "iscrowd",
                "slice_index_x",
                "slice_index_y",
            ],
        ),
    ]

    postprocessor = Postprocessor(
        data_set,
        post_processing_steps,
        progress_bar_description="Simplifying grain boundaries",
    )
    postprocessor.run()


if __name__ == "__main__":
    fire.Fire(measure_grain_properties)
