import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import plot_residuals


def plot_ma_smoothing(axes, series, q=5):
    """
    Performs two sided moving average smoothing.

    :param axes: two axes to plot on
    :param series: np.ndarray with the series
    :param q: int, window of the moving average
    """

    smoothed_series = []
    normalize_constant = 1 / (2*q + 1)
    for i in range(q, len(series) - q):
        smoothed_series.append(normalize_constant * np.sum(series[i - q: i + q + 1]))
    smoothed_series = np.array(smoothed_series)

    x_vals = list(range(len(series)))

    ax = axes[0]
    ax.plot(x_vals, series, label="Sales", color="blue", linestyle="None", marker="o", markersize=2)
    ax.plot(x_vals[q:len(series)-q], smoothed_series, color="orange", label="Two-sided MA")
    ax.set_title("Two-sided MA series with q={}".format(q))
    ax.legend()

    valid_slice = slice(q, len(series) - q)
    plot_residuals(axes[1], x_vals[valid_slice], series[valid_slice], smoothed_series)


def plot_ema_smoothing(axes, series, alpha=0.4):
    """
    Performs exponential smoothing.

    :param axes: two axes to plot on
    :param series: np.ndarray with the series
    :param alpha: float, smoothing operator
    """

    ema_series = [series[1]]
    for i in range(1, len(series)):
        ema_series.append(alpha * series[i] + (1 - alpha) * ema_series[-1])
    ema_series = np.array(ema_series)
    x_vals = list(range(len(series)))
    ax = axes[0]
    ax.plot(x_vals, series, label="Sales", color="blue", linestyle="None", marker="o", markersize=2)
    ax.plot(x_vals, ema_series, color="orange", label="EMA")
    ax.set_title("EMA series with alpha={}".format(alpha))
    ax.legend()

    plot_residuals(axes[1], x_vals, series, ema_series)


def plot_differencing(axes, series, k=2):
    """
    Performs differencing.

    :param axes: two axes to plot on
    :param series: np.ndarray with the series
    :param k: int, differencing operator
    """

    differenced_series = series.copy()
    for _ in range(k):
        differenced_series = differenced_series[1:] - differenced_series[:-1]
    x_vals = list(range(len(series)))
    ax = axes[0]
    ax.plot(x_vals, series, label="Sales", color="blue", linestyle="None", marker="o", markersize=2)
    ax.plot(x_vals[k:], differenced_series, color="orange", label="differenced series")
    ax.set_title("Series {}th-order differenced".format(k))
    ax.legend()


def main():
    fig, axes = plt.subplots(2, 3, sharex=True)
    data = pd.read_csv("../data/shampoo.csv")
    series = data["Sales"].values
    plot_ma_smoothing(axes[:, 0], series, q=2)
    plot_ema_smoothing(axes[:, 1], series, alpha=0.4)
    plot_differencing(axes[:, 2], series, k=2)
    plt.show()

if __name__ == "__main__":
    main()
