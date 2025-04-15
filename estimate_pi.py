"""
Estimate pi via Monte Carlo integration.
"""
import matplotlib.pyplot as plt
import numpy as np


def samplePoints(sample_size: int, rng: np.random.Generator) -> np.ndarray:
    # Generate N random 2d points within the region [-1, 1] x [-1, 1].
    # We can avoid square roots by squaring the radius instead.
    points_2d = 2 * rng.random((sample_size, 2)) - 1
    mc_points = np.uint8(np.sum(points_2d ** 2, axis=1) <= 1)

    return mc_points


def estimatePi(mc_points: np.ndarray):
    # Area of circle is (pi). Area of square is (2 ** 2).
    # The probability that a random point generated in the square lies within the circle is pi / 4.
    # Therefore, in_circle/num_pts should approach pi / 4 as the number of points increases.
    in_circle = np.count_nonzero(mc_points)
    pi_est = 4 * in_circle / len(mc_points)
    mc_err = 4 * np.std(mc_points) / np.sqrt(len(mc_points))

    return pi_est, mc_err


def plotPiEstimate(min_sample: int, max_sample: int, num_points: int, savefig: bool = False,
                   rng: np.random.Generator = np.random.default_rng(seed=42)) -> None:
    """
    Plot the estimate of pi as a function of the sample size.

    Points for the plot are logarithmically spaced with base 10.
    The error estimates are plotted as bands around the estimate for pi.

    Parameters
    ----------
    min_sample : int
        Minimum sample size.
    max_sample : int
        Maximum sample size.
    num_points : int
        Number of points for the plot, including the endpoints.
    savefig : bool
        If true, saves the plot in figures/ex1_pi.png.
    rng : numpy.random.Generator, default=numpy.random.default_rng(seed=42)
        Numpy random number generator to generate random points.

    Returns
    -------
    None
    """
    min_exp = np.log10(min_sample)
    max_exp = np.log10(max_sample)

    sample_sizes = np.logspace(start=min_exp, stop=max_exp, num=num_points, endpoint=True, base=10, dtype=int)
    mc_pts = samplePoints(sample_size=sample_sizes[-1], rng=rng)

    estimates = [estimatePi(mc_pts[:sample_size]) for sample_size in sample_sizes]
    y_values = np.array([estimate[0] for estimate in estimates])
    e_values = np.array([estimate[1] for estimate in estimates])

    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_xlim(0.8 * sample_sizes[0], 1.2 * sample_sizes[-1])
    ax.plot(sample_sizes, y_values, color='royalblue')
    ax.fill_between(sample_sizes, y_values - e_values, y_values + e_values, alpha=0.25, facecolors='dodgerblue',
                    label="Error estimate")
    ax.hlines(np.pi, sample_sizes[0], sample_sizes[-1], colors="r", linestyles='dashed')
    ax.set_xlabel(r"Sample size $N$", fontsize=14)
    ax.set_ylabel(r"Estimate of $\pi$", fontsize=14)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    plt.legend()

    if savefig:
        plt.savefig("figures/ex1_pi.png", dpi=400, bbox_inches='tight')

    plt.show()

    return


if __name__ == '__main__':
    plotPiEstimate(min_sample=10, max_sample=1_000_000, num_points=200, savefig=False)
