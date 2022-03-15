import warnings
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import polygonize, unary_union
from skimage.measure import find_contours

from paddle.custom_types import ArrayLike, Image
from paddle.postprocessing import PostProcessingStepBase


def normalize(vector: np.ndarray) -> np.ndarray:
    return vector / np.sqrt(np.sum(vector ** 2))


def average_points(a: Point, b: Point) -> Point:
    return LineString([a, b]).interpolate(0.5, normalized=True)


def calculate_angle_degree(direction1, direction2) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        angle_degree = np.rad2deg(np.arccos(direction1 @ direction2))

    return angle_degree


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

    @property
    def coordination_number(self) -> Optional[int]:
        if self.dihedral_angles is None:
            return None

        return len(self.dihedral_angles)

    @property
    def dihedral_angles(self) -> Optional[List[float]]:
        if self.boundary_simplified is None:
            return None

        segments = list(
            map(
                LineString,
                zip(
                    self.boundary_simplified.coords[:-1],
                    self.boundary_simplified.coords[1:],
                ),
            )
        )

        angles: List[float] = []

        def _segment_to_direction(segment):
            return normalize(np.diff(segment.xy).squeeze())

        for segment1, segment2 in zip(segments, segments[1:] + [segments[0]]):
            direction1 = _segment_to_direction(segment1)
            direction2 = _segment_to_direction(segment2)

            angle = 180 - calculate_angle_degree(direction1, direction2)

            if angle < 180:
                angles.append(angle)

        return angles

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
    """Simplify the grain boundaries of an image."""

    def __call__(
        self, image: Image, annotation: Dict[str, Any]
    ) -> Tuple[Image, Dict[str, Any]]:
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
        grains = [Grain(mask, mask_id) for mask_id, mask in enumerate(masks)]

        for grain_a in grains:

            for grain_b in grains:
                if do_average_boundaries:
                    if grain_b.id <= grain_a.id:
                        continue

                edge_cut = grain_a.calculate_common_edge_cut(
                    grain_b, do_average_edge=do_average_boundaries
                )

                if edge_cut is not None:
                    grain_a.edge_cuts.append(edge_cut)
                    grain_b.edge_cuts.append(edge_cut)

        for grain in grains:

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

        do_visualization = False

        if do_visualization:

            self._visualization(image, grains)

        # store data in annotation
        polygons: List[Optional[np.ndarray]] = []
        mask_polygons: List[Optional[np.ndarray]] = []
        polygon_areas: List[Optional[float]] = []
        coordination_numbers: List[Optional[int]] = []
        dihedral_angle_lists: List[Optional[List[float]]] = []

        for grain in grains:
            coordination_numbers.append(grain.coordination_number)
            dihedral_angle_lists.append(grain.dihedral_angles)
            mask_polygons.append(np.asarray(grain.boundary_original.xy))
            if grain.boundary_simplified is None:
                polygons.append(None)
                polygon_areas.append(None)
            else:
                polygons.append(np.asarray(grain.boundary_simplified.xy))
                polygon_areas.append(Polygon(grain.boundary_simplified).area)

        annotation["polygons"] = polygons
        annotation["mask_polygons"] = mask_polygons
        annotation["polygon_area"] = polygon_areas
        annotation["polygon_coordination_number"] = coordination_numbers
        annotation["polygon_dihedral_angle"] = dihedral_angle_lists

        return image, annotation

    def _visualization(self, image, grains):
        def _plot_image(image: Image):
            plt.imshow(image, cmap="gray")
            image_height, image_width = image.shape
            plt.axis([0, image_width, 0, image_height])
            plt.axis("off")

        np.random.seed(41)
        num_grains = len(grains)
        colors = plt.cm.viridis(np.random.rand(num_grains))

        example_grain = grains[16]

        # Original boundaries.
        _plot_image(image)
        for grain, color in zip(grains, colors):
            plt.plot(*grain.boundary_original.xy, "-", linewidth=0.5, c=color)
            # plt.annotate(
            #     str(grain.id), np.asarray(grain.center_of_mass.xy)
            # )

        plt.savefig("01boundary_original.png", bbox_inches="tight")
        plt.close()

        # Convex hull boundaries.
        _plot_image(image)
        for grain, color in zip(grains, colors):
            plt.plot(*grain.boundary.xy, "-", linewidth=0.5, c=color)

        plt.savefig("02boundary_convex.png", bbox_inches="tight")
        plt.close()

        # Connections
        _plot_image(image)
        for grain, color in zip(grains, colors):
            plt.plot(*grain.boundary.xy, "-", linewidth=0.5, c=color)

        for connection in example_grain.connections:
            plt.plot(*connection.xy, "-", linewidth=0.5, c="black", marker="o")

        plt.plot(*example_grain.center_of_mass.xy, "o", linewidth=0.5, c="red")

        plt.savefig("03connections.png", bbox_inches="tight")
        plt.close()

        # Edgecut construction
        example_grain2 = grains[13]
        _plot_image(image)
        connection = example_grain.connection_to(example_grain2)
        edgecut = example_grain.calculate_common_edge_cut(example_grain2)

        for grain, color in zip(grains, colors):
            if grain is not example_grain and grain is not example_grain2:
                continue
            plt.plot(*grain.boundary.xy, "-", linewidth=0.5, c=color)

        plt.plot(*connection.xy, "-", linewidth=0.5, c="black", marker="o")
        assert edgecut is not None
        plt.plot(*edgecut.xy, "-", linewidth=0.5, c="red")

        plt.savefig("04edge_cut_construction.png", bbox_inches="tight")
        plt.close()

        # Edgecuts
        _plot_image(image)
        for grain, color in zip(grains, colors):
            plt.plot(*grain.boundary.xy, "-", linewidth=0.5, c=color)

        for edge_cut in example_grain.edge_cuts:
            plt.plot(*edge_cut.xy, "-", linewidth=0.5, c="black")

        plt.plot(*example_grain.center_of_mass.xy, "o", linewidth=0.5, c="red")

        plt.savefig("05edge_cuts.png", bbox_inches="tight")
        plt.close()

        # Final polygons
        _plot_image(image)

        for grain, color in zip(grains, colors):
            if grain.boundary_simplified is None:
                continue

            alpha = 0.75
            color[3] = alpha
            plt.plot(
                *grain.boundary_simplified.xy,
                "-",
                linewidth=0.5,
                c=color,
            )

        plt.savefig("06boundaries_simplified.png", bbox_inches="tight")
        plt.close()
