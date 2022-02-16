from pathlib import Path
from typing import Tuple

import fire
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, spatial

from paddle.custom_types import Annotation, AnyPath, Image
from paddle.data import MaskRCNNDataset
from paddle.postprocessing import Postprocessor
from paddle.postprocessing.postprocessingsteps import (
    FilterScore,
    PostProcessingStepBase,
)

CSV_FILE_NAME = "voronoi_cells.csv"


class ReconstructVoronoiCells(PostProcessingStepBase):
    """Reconstruct the voronoi cells of an image."""

    def __init__(self, output_file_path: AnyPath) -> None:
        self.output_file_path = Path(output_file_path)

        if self.output_file_path.exists():
            raise FileExistsError(f"File already exists: {output_file_path}")

    def __call__(
        self, image: Image, annotation: Annotation
    ) -> Tuple[Image, Annotation]:
        """Restore Voronoi cells based on a set of masks.

        :param image: input image
        :param annotation: Dictionary containing annotation of an image (e.g. masks,
            bounding boxes, etc.)
        :return: image and annotation, both identical to the inputs
        """
        masks = annotation["masks"]
        image = np.mean(image, axis=0)

        # calculate weighted centroids
        centers_of_mass = [
            tuple(reversed(ndimage.measurements.center_of_mass(mask)))
            for mask in masks
        ]

        vor = spatial.Voronoi(centers_of_mass)

        plt.imshow(image, cmap="gray")

        spatial.voronoi_plot_2d(
            vor,
            show_vertices=False,
            line_width=1,
            line_alpha=0.5,
            point_size=4,
            ax=plt.gca(),
        )
        plt.show()

        exit()

        return image, annotation


def reconstruct_voronoi_cells(data_root: AnyPath, subset: str):
    data_root = Path(data_root)

    data_set = MaskRCNNDataset(
        data_root,
        subset=subset,
    )

    subset_folder_path = data_root / subset
    measurement_csv_path = subset_folder_path / CSV_FILE_NAME

    post_processing_steps = [
        FilterScore(0.8),
        ReconstructVoronoiCells(measurement_csv_path),
    ]

    postprocessor = Postprocessor(
        data_set,
        post_processing_steps,
        progress_bar_description="Reconstructing Voronoi cells.",
    )
    # postprocessor.log(subset_folder_path)
    postprocessor.run()


if __name__ == "__main__":
    data_root = "output/ZnO"
    fire.Fire(reconstruct_voronoi_cells)
