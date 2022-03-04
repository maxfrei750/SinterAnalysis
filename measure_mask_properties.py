from pathlib import Path
from typing import List

import fire
import numpy as np
import skimage
from shapely.geometry import Polygon, box
from skimage.measure import find_contours

from paddle.data import MaskRCNNDataset
from paddle.postprocessing import (
    Postprocessor,
    SaveMaskProperties,
    calculate_areas,
    calculate_maximum_feret_diameters,
    calculate_minimum_feret_diameters,
)

CSV_FILE_NAME = "mask_properties.csv"


def calculate_distances_to_image_border(masks: np.ndarray) -> np.ndarray:
    """Calculate distances to image border for a numpy array containing masks.

    :param masks: NxHxW numpy array, which stores N instance masks.
    :return: Numpy array of distances.
    """

    distances: List[float] = []

    for mask in masks:
        image_height, image_width = mask.shape
        image_box = box(0, 0, image_width, image_height).exterior

        mask_outline = Polygon(
            np.squeeze(find_contours(np.transpose(mask), 0)[0])
        ).exterior

        distance = mask_outline.distance(image_box)
        distances.append(distance)

    return np.asarray(distances)


def calculate_minor_axis_lengths(masks: np.ndarray) -> np.ndarray:
    """Calculate minor axis lengths for a numpy array containing masks.

    :param masks: NxHxW numpy array, which stores N instance masks.
    :return: Numpy array of minor axis ratios.
    """

    minor_axis_lengths: List[float] = []

    for mask in masks:
        region_properties = skimage.measure.regionprops(mask)
        minor_axis_lengths.append(region_properties[0].minor_axis_length)

    return np.asarray(minor_axis_lengths)


def calculate_major_axis_lengths(masks: np.ndarray) -> np.ndarray:
    """Calculate major axis lengths for a numpy array containing masks.

    :param masks: NxHxW numpy array, which stores N instance masks.
    :return: Numpy array of major axis ratios.
    """

    major_axis_lengths: List[float] = []

    for mask in masks:
        region_properties = skimage.measure.regionprops(mask)
        major_axis_lengths.append(region_properties[0].major_axis_length)

    return np.asarray(major_axis_lengths)


def calculate_perimeters(masks: np.ndarray) -> np.ndarray:
    """Calculate perimiters for a numpy array containing masks.

    :param masks: NxHxW numpy array, which stores N instance masks.
    :return: Numpy array of perimeters.
    """
    masks = np.array(masks).astype(bool)
    perimeters = np.asarray(
        [skimage.measure.perimeter(mask) for mask in masks]
    )
    return perimeters


def measure_particle_size_distribution(data_root, subset):
    """Measures the particle size distribution (maximum Feret diameter) of a
    dataset.

    :param data_root: Path of the data set folder, holding the subsets.
    :param subset: Name of the subset to use.
    """

    data_root = Path(data_root)

    data_set = MaskRCNNDataset(
        data_root,
        subset=subset,
    )

    subset_folder_path = data_root / subset
    measurement_csv_path = subset_folder_path / CSV_FILE_NAME
    measurement_fcns = {
        "feret_diameter_max": calculate_maximum_feret_diameters,
        "feret_diameter_min": calculate_minimum_feret_diameters,
        "area": calculate_areas,
        "perimeter": calculate_perimeters,
        "axis_length_min": calculate_minor_axis_lengths,
        "axis_length_max": calculate_major_axis_lengths,
        "distance_to_image_border": calculate_distances_to_image_border,
    }

    post_processing_steps = [
        SaveMaskProperties(
            measurement_csv_path,
            measurement_fcns=measurement_fcns,
        ),
    ]

    postprocessor = Postprocessor(
        data_set,
        post_processing_steps,
        progress_bar_description="Measuring mask properties",
    )
    postprocessor.log(subset_folder_path)
    postprocessor.run()


if __name__ == "__main__":
    fire.Fire(measure_particle_size_distribution)
