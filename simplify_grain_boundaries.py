from pathlib import Path
from typing import List, Optional, Tuple

import fire
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import polygonize, unary_union
from skimage.measure import find_contours

from paddle.custom_types import Annotation, AnyPath, ArrayLike, Image
from paddle.data import MaskRCNNDataset
from paddle.postprocessing import Postprocessor
from paddle.postprocessing.postprocessingsteps import (
    FilterScore,
    PostProcessingStepBase,
)

CSV_FILE_NAME = "voronoi_cells.csv"


def normalize(vector: np.ndarray) -> np.ndarray:
    return vector / np.sqrt(np.sum(vector ** 2))


def average_points(a: Point, b: Point) -> Point:
    return LineString([a, b]).interpolate(0.5, normalized=True)


def average_directions(
    direction_a: np.ndarray, direction_b: np.ndarray
) -> np.ndarray:
    # Align direction vectors.
    if direction_a @ direction_b < 0:  # dot product
        direction_b = -direction_b

    return (direction_a + direction_b) / 2


class Grain:
    def __init__(self, mask: ArrayLike, index: int) -> None:
        self.id = index
        self.mask = mask

        self.polygon_original = Polygon(
            np.squeeze(find_contours(np.transpose(mask), 0)[0])
        )

        self.boundary_original = self.polygon_original.exterior

        self.boundary = self.boundary_original.convex_hull.exterior

        self.center_of_mass = Point(
            np.array(self.boundary.centroid.coords.xy).squeeze()
        )

        image_diagonal_length = np.sqrt(sum(np.array(mask.shape) ** 2))
        self._cut_length = image_diagonal_length * 2

        self.boundary_segments = list(
            map(
                LineString,
                zip(self.boundary.coords[:-1], self.boundary.coords[1:]),
            )
        )

        self.edge_cuts: List[LineString] = []
        self.boundary_simplified: Optional[LineString] = None
        self.connections: List[LineString] = []

    def connection_to(self, other_grain: "Grain") -> LineString:
        return LineString([self.center_of_mass, other_grain.center_of_mass])

    def calculate_edge_direction(
        self, intersection_point: Point
    ) -> np.ndarray:
        for segment in self.boundary_segments:
            if intersection_point.distance(segment) < 1:  # 1 pixel
                return normalize(np.diff(segment.xy).squeeze())

        raise RuntimeError("Could not determine edge direction.")

    def calculate_common_edge_cut(
        self, other_grain: "Grain", do_average_edge: bool = True
    ) -> Optional[LineString]:

        connection = self.connection_to(other_grain)
        self.connections.append(connection)
        other_grain.connections.append(connection)

        intersection_point = self.boundary.intersection(connection)
        edge_direction = self.calculate_edge_direction(intersection_point)

        if intersection_point.is_empty:
            return None

        if do_average_edge:
            intersection_point_other = other_grain.boundary.intersection(
                connection
            )

            if (
                intersection_point.is_empty
                or intersection_point_other.is_empty
            ):
                return None

            intersection_point_distance_threshold = 1

            intersection_point_distance = intersection_point.distance(
                intersection_point_other
            )

            if (
                intersection_point_distance / connection.length
            ) > intersection_point_distance_threshold:
                return None

            support_vector = average_points(
                intersection_point, intersection_point_other
            )

            edge_direction_other = other_grain.calculate_edge_direction(
                intersection_point_other
            )

            edge_angle_threshold = 90
            edge_angle = np.rad2deg(
                np.arccos(edge_direction @ edge_direction_other)
            )
            if edge_angle > edge_angle_threshold and edge_angle < (
                180 - edge_angle_threshold
            ):  # degree
                return None

            direction = average_directions(
                edge_direction, edge_direction_other
            )

        else:
            support_vector = intersection_point
            direction = edge_direction

        return self.calculate_edge(support_vector, direction)

    def calculate_edge(
        self, support_vector: Point, direction: np.ndarray
    ) -> LineString:
        p1 = (
            np.array(support_vector.xy).squeeze()
            - direction * self._cut_length / 2
        )
        p2 = (
            np.array(support_vector.xy).squeeze()
            + direction * self._cut_length / 2
        )

        return LineString([p1, p2])


class SimplifyGrainBoundaries(PostProcessingStepBase):
    """Simplify the grain boundaires of an image."""

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
        do_average_boundaries = True

        masks = annotation["masks"]

        assert isinstance(masks, np.ndarray)

        image = np.mean(image, axis=0)
        image_height, image_width = image.shape
        image_box = box(0, 0, image_width, image_height)

        grains = [Grain(mask, mask_id) for mask_id, mask in enumerate(masks)]

        for grain_a in grains:

            for grain_b in grains:

                # if not (grain_id_a == 31 or grain_id_b == 31):
                #     continue

                # if grain_id_b == 0:
                #     plt.plot(
                #         *grain_a.connection_to(grain_b).xy,
                #         "-",
                #         linewidth=0.5,
                #         c="black",
                #     )

                # if grain_a is grain_b:
                #     continue
                if do_average_boundaries:
                    if grain_b.id <= grain_a.id:
                        continue

                edge_cut = grain_a.calculate_common_edge_cut(
                    grain_b, do_average_edge=do_average_boundaries
                )

                if edge_cut is not None:
                    # color_other = np.random.rand(
                    #     3,
                    # )

                    # plt.plot(*edge_cut.xy, "-", linewidth=0.5, c=color_other)

                    # if grain_id_a == 31:
                    #     plt.annotate(
                    #         str(grain_id_b),
                    #         np.mean(np.array(edge_cut.xy), axis=1),
                    #         c=color_other,
                    #     )

                    # if grain_id_b == 31:
                    #     plt.annotate(
                    #         str(grain_id_a),
                    #         np.mean(np.array(edge_cut.xy), axis=1),
                    #         c=color_other,
                    #     )

                    # edge_cut = image_box.intersection(edge_cut)

                    grain_a.edge_cuts.append(edge_cut)
                    grain_b.edge_cuts.append(edge_cut)

        # for edge_cut in grains[12].edge_cuts:
        #     plt.plot(
        #         *edge_cut.xy,
        #         "-",
        #         linewidth=0.5,
        #         c=color,
        #     )

        for grain in grains:

            # if grain.id != 31:
            #     continue

            candidates = polygonize(unary_union(grain.edge_cuts))

            relevant_candidates: List[Polygon] = []

            for candidate in candidates:
                relative_overlap = (
                    candidate.intersection(grain.polygon_original).area
                    / candidate.area
                )

                overlap_threshold = 0.5
                if relative_overlap > overlap_threshold:
                    relevant_candidates.append(candidate)

            if relevant_candidates:
                grain.boundary_simplified = unary_union(
                    relevant_candidates
                ).convex_hull.exterior
            else:
                grain.boundary_simplified = None

            alpha = 1
            color = np.random.rand(
                4,
            )
            color[3] = alpha

            # if grain.boundary_simplified is not None:

            #     plt.plot(
            #         *grain.boundary_simplified.xy,
            #         "-",
            #         linewidth=0.5,
            #         c=color,
            #     )

            a = 1

            # plt.plot(
            #     *grain_a.boundary.xy,
            #     "-",
            #     linewidth=1,
            #     c=color,
            # )

        do_visualization = True

        if do_visualization:

            def _plot_image(image: Image):
                plt.imshow(image, cmap="gray")
                image_height, image_width = image.shape
                plt.axis([0, image_width, 0, image_height])
                plt.axis("off")

            np.random.seed(42)
            num_grains = len(grains)
            colors = plt.cm.viridis(np.random.rand(num_grains))

            example_grain = grains[15]

            # Original boundaries.
            _plot_image(image)
            for grain, color in zip(grains, colors):
                plt.plot(
                    *grain.boundary_original.xy, "-", linewidth=0.5, c=color
                )

            plt.savefig("01boundary_original.png")
            plt.close()

            # Convex hull boundaries.
            _plot_image(image)
            for grain, color in zip(grains, colors):
                plt.plot(*grain.boundary.xy, "-", linewidth=0.5, c=color)

            plt.savefig("02boundary_convex.png")
            plt.close()

            # Connections
            _plot_image(image)
            for grain, color in zip(grains, colors):
                plt.plot(*grain.boundary.xy, "-", linewidth=0.5, c=color)

            for connection in example_grain.connections:
                plt.plot(
                    *connection.xy, "-", linewidth=0.5, c="black", marker="o"
                )

            plt.plot(
                *example_grain.center_of_mass.xy, "o", linewidth=0.5, c="red"
            )

            plt.savefig("03connections.png")
            plt.close()

            # Edgecut construction
            example_grain2 = grains[12]
            _plot_image(image)
            connection = example_grain.connection_to(example_grain2)
            edgecut = example_grain.calculate_common_edge_cut(example_grain2)

            for grain, color in zip(grains, colors):
                if grain is not example_grain and grain is not example_grain2:
                    continue
                plt.plot(*grain.boundary.xy, "-", linewidth=0.5, c=color)

            plt.plot(*connection.xy, "-", linewidth=0.5, c="black", marker="o")
            plt.plot(*edgecut.xy, "-", linewidth=0.5, c="red")

            plt.savefig("04edge_cut_construction.png")
            plt.close()

            # Edgecuts
            _plot_image(image)
            for grain, color in zip(grains, colors):
                plt.plot(*grain.boundary.xy, "-", linewidth=0.5, c=color)

            for edge_cut in example_grain.edge_cuts:
                plt.plot(*edge_cut.xy, "-", linewidth=0.5, c="black")

            plt.plot(
                *example_grain.center_of_mass.xy, "o", linewidth=0.5, c="red"
            )

            plt.savefig("05edge_cuts.png")
            plt.close()

            # Final polygons
            _plot_image(image)
            # for grain in grains:
            #     plt.plot(
            #         *grain.boundary_original.xy, "-", linewidth=0.5, c="black"
            #     )

            for grain, color in zip(grains, colors):
                if grain.boundary_simplified is None:
                    continue
                plt.plot(
                    *grain.boundary_simplified.xy,
                    "--",
                    linewidth=0.5,
                    c=color,
                )

            plt.savefig("06boundaries_simplified.png")
            plt.close()

            exit()

        return image, annotation


def simplify_grain_boundaries(data_root: AnyPath, subset: str):
    data_root = Path(data_root)

    data_set = MaskRCNNDataset(
        data_root,
        subset=subset,
    )

    subset_folder_path = data_root / subset
    measurement_csv_path = subset_folder_path / CSV_FILE_NAME

    post_processing_steps = [
        FilterScore(0.8),
        SimplifyGrainBoundaries(measurement_csv_path),
    ]

    postprocessor = Postprocessor(
        data_set,
        post_processing_steps,
        progress_bar_description="Simplifying grain boundaries",
    )
    # postprocessor.log(subset_folder_path)
    postprocessor.run()


if __name__ == "__main__":
    data_root = "output/ZnO"
    fire.Fire(simplify_grain_boundaries)
