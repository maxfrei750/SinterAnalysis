import pickle
from pathlib import Path
from typing import List

import fire
import numpy as np
from matplotlib import pyplot as plt

from paddle.custom_types import AnyPath


def example_evaluation(
    data_root: AnyPath,
    subset: str,
    scale_meters_per_pixel: float = 1.8867924528301888e-09,
    score_threshold: float = 0.8,
    minimum_border_distance_threshold_pixel: int = 10,
) -> None:
    """Example to show how to evaluate the measured grain properties, using previously
    generated pkl-files.

    Args:
        data_root (AnyPath): Root directory of the data.
        subset (str): Name of the subset folder in the data_root.
        scale_meters_per_pixel (float, optional): Scale [m/pixel], to convert pixels to
            meters. Defaults to 1.8867924528301888e-09 (Jeol 7500F at 50kx).
        score_threshold (float, optional): Measurements with a detection score below
            this threshold are discarded. Defaults to 0.5.
        minimum_border_distance_threshold_pixel (int, optional): Masks that are closer
            to the image border than this threshold are discarded, because, most likely,
            they belong to incomplete grains.
    """

    scale_nanometers_per_pixel = scale_meters_per_pixel / 1e-9

    # Get a list of the pickle files.
    pkl_file_root = Path(data_root) / subset
    pkl_file_paths = sorted(list(pkl_file_root.glob("*.pkl")))

    # Iterate the pickle files.
    for pkl_file_path in pkl_file_paths:

        # Load data.
        with open(pkl_file_path, "rb") as pkl_file:
            data = pickle.load(pkl_file)

        mask_areas_pixel = np.asarray(data["mask_area"])

        # Some polygon areas can be None, if the grain boundary simplification failed.
        # `dtype=float` converts `None` to NaN (not a number), so that we can use the
        # resulting array for calculations.
        polygon_areas_pixel = np.asarray(data["polygon_area"], dtype=float)
        scores = np.asarray(data["scores"])

        # Some arrays from the data are ragged (i.e. non-uniform number of columns per
        # row). With `dtype=object` this is no problem.
        dihedral_angles_list_degree = np.asarray(
            data["polygon_dihedral_angle"], dtype=object
        )

        feret_diameters_max_pixel = np.asarray(data["mask_feret_diameter_max"])

        # Remove measurements with masks too close to the border.
        mask_distances_to_image_border_pixel = data[
            "mask_distance_to_image_border"
        ]
        is_too_close_to_image_border = (
            mask_distances_to_image_border_pixel
            <= minimum_border_distance_threshold_pixel
        )

        mask_areas_pixel = mask_areas_pixel[~is_too_close_to_image_border]
        polygon_areas_pixel = polygon_areas_pixel[
            ~is_too_close_to_image_border
        ]
        scores = scores[~is_too_close_to_image_border]

        dihedral_angles_list_degree = dihedral_angles_list_degree[
            ~is_too_close_to_image_border
        ]
        feret_diameters_max_pixel = feret_diameters_max_pixel[
            ~is_too_close_to_image_border
        ]

        # Remove measurements below the score_threshold.
        is_relevant_score = scores >= score_threshold

        mask_areas_pixel = mask_areas_pixel[is_relevant_score]
        polygon_areas_pixel = polygon_areas_pixel[is_relevant_score]
        scores = scores[is_relevant_score]

        dihedral_angles_list_degree = dihedral_angles_list_degree[
            is_relevant_score
        ]
        feret_diameters_max_pixel = feret_diameters_max_pixel[
            is_relevant_score
        ]

        # Error of the polygon areas, as proposed by M. W.
        absolute_errors_percent = np.abs(
            (mask_areas_pixel - polygon_areas_pixel) / mask_areas_pixel
        )

        polygon_area_mean_absolute_percentage_error = np.average(
            absolute_errors_percent[~np.isnan(absolute_errors_percent)],
            weights=scores[~np.isnan(absolute_errors_percent)],
        )  # only use absolute_percentage_errors (and associated scores) that are not NaNs.

        # Plot dihedral angle histogram (just one image).

        # Since there are multiple angles per instance (i.e. per score), we need to
        # repeat the score of each instance n times, where n is the number of angles,
        # to use the scores as weights for the histogram:
        scores_dihedral_angles_degree: List[float] = []
        dihedral_angles_degree: List[float] = []
        for score, dihedral_angles_per_instance_degree in zip(
            scores, dihedral_angles_list_degree
        ):
            if dihedral_angles_per_instance_degree is None:
                continue

            num_angles = len(dihedral_angles_per_instance_degree)
            scores_dihedral_angles_degree += [score] * num_angles
            dihedral_angles_degree += dihedral_angles_per_instance_degree

        plt.hist(
            dihedral_angles_degree,  # Concatenate the ragged array.
            weights=scores_dihedral_angles_degree,
            density=True,
        )

        plt.xlim([0, 180])
        plt.xlabel("Dihedral angle/°")
        plt.ylabel("Probability density")
        plt.title("Dihedral angle histogram")
        plt.show()

        # Plot maximum Feret diameter histogram.

        plt.hist(
            feret_diameters_max_pixel * scale_nanometers_per_pixel,
            weights=scores,
            density=True,
        )

        plt.xlabel("Feret diameter/nm")
        plt.ylabel("Probability density")
        plt.title("Maximum Feret diameter histogram")
        plt.show()

        # Print MAPE of the polygon area
        print(
            f"Mean absolute percentage error of the polygon area: "
            f"{polygon_area_mean_absolute_percentage_error: .1%}"
        )

        # Plot polygons.
        image_path = pkl_file_root / f"image_{data['image_name']}.png"
        image = plt.imread(image_path)
        plt.imshow(image, cmap="gray")
        for polygon, score, mask_distances_to_image_border_pixel in zip(
            data["polygons"],
            data["scores"],
            data["mask_distance_to_image_border"],
        ):
            if polygon is None:
                continue

            if score < score_threshold:
                continue

            if (
                mask_distances_to_image_border_pixel
                < minimum_border_distance_threshold_pixel
            ):
                continue

            color = plt.cm.get_cmap("hsv")(np.random.rand())

            plt.plot(polygon[0, :], polygon[1, :], "-", linewidth=0.5, c=color)
        image_height, image_width, _ = image.shape
        plt.axis([0, image_width, 0, image_height])
        plt.gca().invert_yaxis()
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    fire.Fire(example_evaluation)
