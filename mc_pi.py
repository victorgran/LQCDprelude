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


def plotEstimate(rng: np.random.Generator, min_sample: int, max_sample: int, num_points: int,
                 base: int | float = 10, endpoint: bool = True) -> None:
    if base != 10:
        min_exp = np.log10(min_sample) / np.log10(base)
        max_exp = np.log10(max_sample) / np.log10(base)
    else:
        min_exp = np.log10(min_sample)
        max_exp = np.log10(max_sample)

    sample_sizes = np.logspace(start=min_exp, stop=max_exp, num=num_points,
                               endpoint=endpoint, base=base, dtype=int)
    mc_pts = samplePoints(sample_size=sample_sizes[-1], rng=rng)

    estimates = [estimatePi(mc_pts[:sample_size]) for sample_size in sample_sizes]
    y_values = [estimate[0] for estimate in estimates]
    e_values = [estimate[1] for estimate in estimates]

    # TODO: Plot the estimation for pi as a function of the sample size.
    # TODO: Add errors to the previous plot.
    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_xlim(0.8 * sample_sizes[0], 1.2 * sample_sizes[-1])
    ax.errorbar(sample_sizes, y_values, yerr=e_values)
    ax.hlines(np.pi, sample_sizes[0], sample_sizes[-1], colors="r")
    plt.show()

    return


def main(sample_size: int, seed: int) -> None:
    rng = np.random.default_rng(seed=seed)  # Seed the r.n.g. for reproducibility.
    plotEstimate(rng=rng, min_sample=10, max_sample=10_000, num_points=100)
    # mc_pts = samplePoints(sample_size, rng)
    # pi_est, mc_err = estimatePi(mc_pts)
    # print(f"{pi_est} +/- {mc_err}")

    return


if __name__ == '__main__':
    main(sample_size=100, seed=42)
    # main(num_points=100_000_000, seed=42)
