from pathlib import Path

import fire

from paddle.data import MaskRCNNDataset
from paddle.postprocessing import (
    Postprocessor,
    SaveMaskProperties,
    calculate_area_equivalent_diameters,
    calculate_maximum_feret_diameters,
)

CSV_FILE_NAME = "particle_size_distribution.csv"


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
        "area_equivalent_diameter": calculate_area_equivalent_diameters,
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
        progress_bar_description="Measuring particle size distribution",
    )
    postprocessor.log(subset_folder_path)
    postprocessor.run()


if __name__ == "__main__":
    fire.Fire(measure_particle_size_distribution)
