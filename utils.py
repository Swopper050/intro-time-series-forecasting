import matplotlib.pyplot as plt
import numpy as np


def get_iid_noise_series(*, length=1000):
    """
    :param length: int, length of the time series
    :returns: np.ndarray with iid noise
    """

    return np.random.rand(length) * 2. - 1.


def ACVF(series, *, h):
    """
    Calculates the sample autocovariance for a given series.

    :param series: np.ndarray with the time series
    :param h: int, lag
    :returns: autocovariance value
    """

    mean = np.mean(series)
    centered_series = series - mean
    if h == 0:
        return (1. / series.shape[0]) * np.sum(centered_series * centered_series)
    else:
        return (1. / series.shape[0]) * np.sum(centered_series[:-h] * centered_series[h:])


def ACF(series, *, h):
    """
    Calculates the sample autocorrelation for a given series.

    :param series: np.ndarray, the time series
    :param h: int, lag
    :returns: autocorrelation value
    """
    return ACVF(series, h=h) / ACVF(series, h=0)


def plot_ACVF(series, *, max_lag):
    """
    Plots autocovariance values up to a specified number of lags plus bounds between
    which most should fall.

    :param series: np.ndarray with values
    :param max_lag: int, max lag value
    """

    fig, (ax_series, ax_acvf) = plt.subplots(nrows=2)
    acvf_vals = [ACVF(series, h=lag) for lag in range(max_lag)]
    acvf_bound = 1.96 / np.sqrt(series.shape[0])

    ax_series.plot(series)
    ax_series.axhline(y=0, xmin=0, xmax=series.shape[0], linestyle="--", color="grey", alpha=0.5)
    ax_series.set_title("IID noise time series")

    ax_acvf.bar(list(range(max_lag)), acvf_vals)
    ax_acvf.axhline(y=0, xmin=0, xmax=max_lag, color="black")
    ax_acvf.axhline(y=acvf_bound, xmin=0, xmax=max_lag, color="grey", linestyle="--", alpha=0.5)
    ax_acvf.axhline(y=-acvf_bound, xmin=0, xmax=max_lag, color="grey", linestyle="--", alpha=0.5)
    ax_acvf.set_title("Autocovariance values for the time series for various lags")

    plt.show()
