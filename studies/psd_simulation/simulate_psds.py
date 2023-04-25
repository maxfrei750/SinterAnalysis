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
    measurements_true: List[float]
    measurement_standard_deviation: float
    random_seed: int = 0

    def __post_init__(self):
        self.random_number_generator = np.random.RandomState(self.random_seed)
        self.measurements_biased = [
            self.random_number_generator.normal(
                loc=m, scale=m * self.measurement_standard_deviation
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

    def plot(self, **kwargs):
        """Plot the measured particle size distribution."""
        sns.kdeplot(self.measurements_biased, clip=(0, float("inf")), **kwargs)


@dataclasses.dataclass
class TrueParticleSizeDistribution:
    geometric_standard_deviation: float
    geometric_mean_diameter: float

    def __post_init__(self):
        self.distribution = scipy_stats.lognorm(
            s=np.log(self.geometric_standard_deviation),
            scale=self.geometric_mean_diameter,
        )

    def plot(self, **kwargs):
        """Plot the accurate particle size distribution."""
        x = np.logspace(-2, 2, num=1000)
        plt.plot(x, self.distribution.pdf(x), **kwargs)

    def measure(
        self,
        num_samples: int,
        measurement_standard_deviation: float,
        random_seed: int = 0,
    ) -> MeasuredParticleSizeDistribution:
        """Spawn a measurement of the particle size distribution."""
        return MeasuredParticleSizeDistribution(
            measurements_true=self.distribution.rvs(
                size=num_samples, random_state=random_seed
            ),
            measurement_standard_deviation=measurement_standard_deviation,
            random_seed=random_seed,
        )


def plot_psd_comparison(
    TrueParticleSizeDistribution,
    geometric_mean_diameter,
    geometric_standard_deviation,
    num_samples,
    measurement_errors,
):
    psd_true = TrueParticleSizeDistribution(
        geometric_mean_diameter=geometric_mean_diameter,
        geometric_standard_deviation=geometric_standard_deviation,
    )
    psd_true.plot(
        marker="",
        label=rf"True PSD ($d_g={geometric_mean_diameter}$, $\sigma_g={geometric_standard_deviation}$)",
        color="gray",
        linestyle="--",
    )

    colors = list(sns.color_palette("viridis", n_colors=len(measurement_errors)))  # type: ignore

    for measurement_error, color in zip(measurement_errors, colors):
        psd_measured = psd_true.measure(
            num_samples=num_samples,
            measurement_standard_deviation=measurement_error,
        )

        error_geometric_mean_diameter_percent = (
            (psd_measured.geometric_mean_diameter - geometric_mean_diameter)
            / psd_true.geometric_mean_diameter
            * 100
        )
        error_geometric_standard_deviation_percent = (
            (
                psd_measured.geometric_standard_deviation
                - psd_true.geometric_standard_deviation
            )
            / geometric_standard_deviation
            * 100
        )

        psd_measured.plot(
            marker="",
            label=rf"Simulated Measurement ($N={num_samples}$, $s={measurement_error*100:.0f}\%$)"
            + "\n"
            + rf"    $\Delta d_g={error_geometric_mean_diameter_percent:.0f}\%$, $\Delta\sigma_g={error_geometric_standard_deviation_percent:.0f}\%$",
            color=color,
        )
    plt.xlabel("Particle Diameter")
    plt.legend()
    plt.xlim(left=0, right=150)


if __name__ == "__main__":
    np.random.seed(0)
    geometric_mean_diameter = 20
    geometric_standard_deviation = 1.5
    measurement_errors = np.arange(0, 0.4, 0.1)

    nums_samples = [1000, 10000, 100000]


    for num_samples in nums_samples:
        fig = plt.figure(figsize=(6, 4))
        plot_psd_comparison(
            TrueParticleSizeDistribution,
            geometric_mean_diameter,
            geometric_standard_deviation,
            num_samples,
            measurement_errors,
        )
        plt.savefig(SCRIPT_DIR / f"psd_simulation_{num_samples}.pdf")
        plt.close()