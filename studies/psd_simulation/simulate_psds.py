"""This module provides classes and functions to simulate and plot particle size
distributions. The module defines two data classes: TrueParticleSizeDistribution and
MeasuredParticleSizeDistribution. The former defines a true particle size distribution
based on its geometric mean diameter and geometric standard deviation, and the latter
simulates a measured particle size distribution based on the true distribution and a
specified measurement standard deviation. The module also provides a function,
plot_psd_comparison, which generates a plot comparing a true particle size distribution
and multiple simulated measured particle size distributions. The function plots each
distribution with its corresponding measurement parameters and error values.

Classes:
    TrueParticleSizeDistribution: A class representing a true particle size
        distribution.
    MeasuredParticleSizeDistribution: A class representing a simulated measured particle
        size distribution.

Functions:
    plot_psd_comparison: A function that generates a plot comparing a true particle size
        distribution and multiple simulated measured particle size distributions.

Constants:
    SCRIPT_DIR: A pathlib.Path object representing the directory of the current script.
"""

import dataclasses
from pathlib import Path
from typing import List

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats as scipy_stats

SCRIPT_DIR = Path(__file__).parent


@dataclasses.dataclass
class MeasuredParticleSizeDistribution:
    """
    Measured particle size distribution.

    Attributes:
        measurements_true (List[float]): List of true measurements.
        measurement_standard_deviation (float): Measurement standard deviation.
        random_seed (int): Random seed for measurement.
    """

    measurements_true: List[float]
    measurement_standard_deviation: float
    random_seed: int = 0

    def __post_init__(self) -> None:
        """
        Post-init method.

        Initializes class attributes and calculates geometric mean diameter and
        geometric standard deviation.
        """
        self.random_number_generator = np.random.RandomState(self.random_seed)
        self.measurements_biased = [
            self.random_number_generator.normal(
                loc=m, scale=self.measurement_standard_deviation
            )
            for m in self.measurements_true
        ]

        # Avoid negative values
        self.measurements_biased = [
            m for m in self.measurements_biased if m > 0
        ]

        self.geometric_mean_diameter = scipy_stats.mstats.gmean(
            self.measurements_biased
        )
        self.geometric_standard_deviation = scipy_stats.gstd(
            self.measurements_biased
        )

    def plot(self, **kwargs) -> None:
        """
        Plot the measured particle size distribution.
        """
        sns.kdeplot(self.measurements_biased, clip=(0, float("inf")), **kwargs)


@dataclasses.dataclass
class TrueParticleSizeDistribution:
    """
    True particle size distribution.

    Attributes:
        geometric_standard_deviation (float): Geometric standard deviation.
        geometric_mean_diameter (float): Geometric mean diameter.
    """

    geometric_standard_deviation: float
    geometric_mean_diameter: float

    def __post_init__(self) -> None:
        """
        Post-init method.

        Initializes the log-normal distribution with given geometric standard deviation
        and geometric mean diameter.
        """
        self.distribution = scipy_stats.lognorm(
            s=np.log(self.geometric_standard_deviation),
            scale=self.geometric_mean_diameter,
        )

    def plot(self, **kwargs) -> None:
        """
        Plot the accurate particle size distribution.
        """
        x = np.logspace(-2, 2, num=1000)
        plt.plot(x, self.distribution.pdf(x), **kwargs)

    def measure(
        self,
        num_samples: int,
        measurement_standard_deviation: float,
        random_seed: int = 0,
    ) -> MeasuredParticleSizeDistribution:
        """
        Spawn a measurement of the particle size distribution.

        Args:
            num_samples (int): Number of samples.
            measurement_standard_deviation (float): Measurement standard deviation.
            random_seed (int): Random seed.

        Returns:
            (MeasuredParticleSizeDistribution): Measured particle size distribution.
        """
        return MeasuredParticleSizeDistribution(
            measurements_true=self.distribution.rvs(
                size=num_samples, random_state=random_seed
            ),
            measurement_standard_deviation=measurement_standard_deviation,
            random_seed=random_seed,
        )


def plot_psd_comparison(
    geometric_mean_diameter_true: float,
    geometric_standard_deviation_true: float,
    num_samples: int,
    measurement_standard_deviations_percent: List[float],
    random_seed=0,
) -> None:
    """Plot the comparison between a true particle size distribution and a simulated
    measured particle size distribution. The true distribution is defined by its
    geometric mean diameter and geometric standard deviation. The function simulates the
    measurement of the particle size distribution by generating a biased distribution
    with a normal distribution and adding noise based on the specified measurement
    standard deviation. The function generates multiple simulated measurements with
    different measurement standard deviations, and for each measurement, it calculates
    the error in the estimated geometric mean diameter and geometric standard deviation
    relative to the true values. Finally, the function plots the true particle size
    distribution and the simulated measurements, labeling each plot with the measurement
    parameters and the calculated errors.

    Args:
        geometric_mean_diameter_true (float): The geometric mean diameter of the true
            particle size distribution in pixels.
        geometric_standard_deviation_true (float): The geometric standard deviation of
            the true particle size distribution.
        num_samples (int): The number of samples to generate for each simulated
            measurement.
        measurement_standard_deviations_percent (List[float]): A list of percentage
            values representing the measurement standard deviation to use for each
            simulated measurement, expressed as a percentage of the geometric mean
            diameter of the true distribution.
        random_seed (int, optional): The random seed to use for generating random
            numbers. Default is 0.

    Returns:
        None
    """

    psd_true = TrueParticleSizeDistribution(
        geometric_mean_diameter=geometric_mean_diameter_true,
        geometric_standard_deviation=geometric_standard_deviation_true,
    )
    psd_true.plot(
        marker="",
        label=rf"True PSD ($d_g={geometric_mean_diameter_true}$ px,"
        + rf" $\sigma_g={geometric_standard_deviation_true}$)",
        color="gray",
        linestyle="--",
    )

    num_simulations = len(measurement_standard_deviations_percent)
    colors = list(sns.color_palette("viridis", n_colors=num_simulations))  # type: ignore

    for measurement_standard_deviation_percent, color in zip(
        measurement_standard_deviations_percent, colors
    ):
        measurement_standard_deviation = (
            measurement_standard_deviation_percent
            * geometric_mean_diameter_true
        )

        psd_measured = psd_true.measure(
            num_samples=num_samples,
            measurement_standard_deviation=measurement_standard_deviation,
            random_seed=random_seed,
        )

        error_geometric_mean_diameter_percent = (
            (
                psd_measured.geometric_mean_diameter
                - geometric_mean_diameter_true
            )
            / psd_true.geometric_mean_diameter
            * 100
        )
        error_geometric_standard_deviation_percent = (
            (
                psd_measured.geometric_standard_deviation
                - psd_true.geometric_standard_deviation
            )
            / geometric_standard_deviation_true
            * 100
        )

        psd_measured.plot(
            marker="",
            label=rf"Simulated Measurement ($N={num_samples}$,"
            + rf" $s={measurement_standard_deviation:.0f}$ px)"
            + "\n"
            + rf"    $\Delta d_g={error_geometric_mean_diameter_percent:.0f}\%$,"
            + rf" $\Delta\sigma_g={error_geometric_standard_deviation_percent:.0f}\%$",
            color=color,
        )
    plt.xlabel("Particle Diameter [px]")
    plt.ylabel("Probability Density [1/px]")
    plt.legend()
    plt.xlim(
        left=0,
        right=psd_true.geometric_mean_diameter
        * psd_true.geometric_standard_deviation
        * 4,
    )


if __name__ == "__main__":
    random_seed = 1
    geometric_mean_diameter_true = 100
    geometric_standard_deviation_true = 1.3
    measurement_standard_deviation = [0, 0.02, 0.04, 0.08]

    nums_samples = [1000, 1000000]

    for num_samples in nums_samples:
        fig = plt.figure(figsize=(6, 4))
        plot_psd_comparison(
            geometric_mean_diameter_true,
            geometric_standard_deviation_true,
            num_samples,
            measurement_standard_deviation,
            random_seed=random_seed,
        )

        output_path = SCRIPT_DIR / f"psd_simulation_{num_samples}.pdf"
        plt.savefig(output_path)
        plt.savefig(output_path.with_suffix(".png"), dpi=300)
        plt.close()
