from pathlib import Path
from typing import Tuple

import fire
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import (
    approximate_polygon,
    find_contours,
    subdivide_polygon,
)
from skimage.morphology.convex_hull import convex_hull_image

from paddle.custom_types import Annotation, AnyPath, Image
from paddle.data import MaskRCNNDataset
from paddle.postprocessing import Postprocessor
from paddle.postprocessing.postprocessingsteps import (
    FilterBorderInstances,
    FilterScore,
    PostProcessingStepBase,
)

CSV_FILE_NAME = "polygon_fits.csv"


class FitPolygons(PostProcessingStepBase):
    """Reconstruct the voronoi cells of an image."""

    def __init__(self, output_file_path: AnyPath) -> None:
        self.output_file_path = Path(output_file_path)

        if self.output_file_path.exists():
            raise FileExistsError(f"File already exists: {output_file_path}")

    def __call__(
        self, image: Image, annotation: Annotation
    ) -> Tuple[Image, Annotation]:
        """Fit each mask with a polygon with as little corners as possible, while still maintaining the shape.

        :param image: input image
        :param annotation: Dictionary containing annotation of an image (e.g. masks,
            bounding boxes, etc.)
        :return: image and annotation, both identical to the inputs
        """
        masks = annotation["masks"]
        image = np.mean(image, axis=0)

        plt.imshow(image, cmap="gray")

        for mask in masks:

            mask = convex_hull_image(mask)

            for contour in find_contours(mask, 0):

                color = np.random.rand(
                    3,
                )

                alpha = 0.5

                plt.plot(
                    contour[:, 1],
                    contour[:, 0],
                    ":",
                    linewidth=1,
                    c=np.append(color * 0.25, alpha),
                )

                coords = approximate_polygon(contour, tolerance=15)
                plt.plot(
                    coords[:, 1],
                    coords[:, 0],
                    "-",
                    linewidth=1,
                    c=np.append(color, alpha),
                )

        plt.show()

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
        FilterBorderInstances(),
        FilterScore(0.8),
        FitPolygons(measurement_csv_path),
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
