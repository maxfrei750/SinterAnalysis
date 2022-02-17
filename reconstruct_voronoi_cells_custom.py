from pathlib import Path
from typing import Tuple

import fire
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import polygonize, split
from skimage.measure import find_contours
from skimage.morphology.convex_hull import convex_hull_image

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

        plt.imshow(image, cmap="gray")

        do_skip_hopping_connections = (
            False  # Looks good, but probably useless.
        )
        do_align_grain_boundaries = True

        contours = [
            Polygon(np.squeeze(find_contours(np.transpose(mask), 0)[0]))
            for mask in masks
        ]

        for contour in contours:
            plt.plot(
                *contour.exterior.xy,
                "-",
                linewidth=0.5,
                c="black",
            )

        contours = [contour.convex_hull for contour in contours]

        for contour in contours:
            plt.plot(
                *contour.exterior.xy,
                "-",
                linewidth=0.5,
                c="black",
            )

        # calculate weighted centroids
        centers_of_mass = np.array(
            [contour.centroid.coords.xy for contour in contours]
        ).squeeze()

        voronoi_data = spatial.Voronoi(centers_of_mass)

        # plt.imshow(masks[7, :, :], cmap="gray")

        # plt.scatter(centers_of_mass[:, 0], centers_of_mass[:, 1])

        image_height, image_width = image.shape

        image_border = box(0, 0, image_width, image_height)

        # plt.plot(*image_polygon.exterior.xy)
        # plt.show()

        image_diagonal_length = np.sqrt(sum(np.array(image.shape) ** 2))
        edge_length = image_diagonal_length * 2

        for grain_id, (contour, center_of_mass) in enumerate(
            zip(contours, centers_of_mass)
        ):

            # TODO: Idea: Forget Voronoi... just iterate over all other grains.

            new_grain = image_border

            for grain_connection_point_ids in voronoi_data.ridge_points:
                if grain_id not in grain_connection_point_ids:
                    continue

                connection_points = [
                    voronoi_data.points[i] for i in grain_connection_point_ids
                ]

                # # Make sure that connections point outwards.
                # if not all(connection_points[0] == center_of_mass):
                #     connection_points = list(reversed(connection_points))

                connection = LineString(connection_points)

                if do_skip_hopping_connections:
                    # Skip connections that intersect more than two contours.
                    num_intersections = 0
                    for c in contours:
                        if not c.exterior.intersection(connection).is_empty:
                            num_intersections += 1

                    if num_intersections > 2:
                        plt.plot(
                            *connection.xy,
                            "-",
                            linewidth=0.5,
                            c="red",
                        )

                        continue

                plt.plot(
                    *connection.xy,
                    "-",
                    linewidth=0.5,
                    c="black",
                )

                intersection_point = contour.exterior.intersection(connection)

                if intersection_point.is_empty:
                    continue

                def segments(curve):
                    return list(
                        map(
                            LineString,
                            zip(curve.coords[:-1], curve.coords[1:]),
                        )
                    )

                def normalize(vector):
                    return vector / np.sqrt(np.sum(vector ** 2))

                def calculate_edge_direction(contour, intersection_point):
                    for segment in segments(contour.exterior):
                        if intersection_point.distance(segment) < 1:
                            return normalize(np.diff(segment.xy).squeeze())

                direction = calculate_edge_direction(
                    contour, intersection_point
                )

                if do_align_grain_boundaries:
                    grain_id_partner = int(
                        grain_connection_point_ids[
                            ~(grain_connection_point_ids == grain_id)
                        ]
                    )

                    contour_partner = contours[grain_id_partner]

                    intersection_point_partner = (
                        contour_partner.exterior.intersection(connection)
                    )

                    if not intersection_point_partner.is_empty:
                        intersection_point = LineString(
                            [intersection_point, intersection_point_partner]
                        ).interpolate(0.5, normalized=True)

                        direction_partner = calculate_edge_direction(
                            contour_partner, intersection_point_partner
                        )

                        # Align direction vectors.
                        if direction @ direction_partner < 0:  # dot product
                            direction_partner = -direction_partner

                        direction = (direction + direction_partner) / 2

                p1 = np.array(intersection_point.xy).squeeze() - direction * edge_length / 2
                p2 = np.array(intersection_point.xy).squeeze() + direction * edge_length / 2

                edge = LineString([p1, p2])

                unioned = new_grain.boundary.union(edge)

                candidates = polygonize(unioned)

                new_grain = [
                    candidate
                    for candidate in candidates
                    if contour.centroid.within(candidate)
                ][0]

            color = np.random.rand(
                3,
            )

            plt.plot(
                *new_grain.exterior.xy,
                ":",
                linewidth=1,
                c=color,
            )

            # plt.plot(
            #     *contour.exterior.xy,
            #     "-",
            #     linewidth=0.5,
            #     c=color,
            # )

            # break

        plt.axis([0, image_width, 0, image_height])
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
