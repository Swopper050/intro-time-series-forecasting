import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def linear_trend(df, ax_main, ax_residuals):
    times = np.arange(df.shape[0]).reshape(df.shape[0], 1)
    X = np.column_stack((np.ones((df.shape[0], 1)), times))
    y = df["Sales"].values.reshape(df.shape[0], 1)

    pseudo_inv = np.dot(np.linalg.inv(np.dot(X.T, X)), X.T)
    solution = np.dot(pseudo_inv, y)
    x0, x1 = solution[0, 0], solution[1, 0]

    sol_line_x = np.arange(df.shape[0])
    sol_line_y = x0 + x1 * sol_line_x

    ax_main.plot(sol_line_x, df["Sales"].values, color="blue", label="sales")
    ax_main.plot(sol_line_x, sol_line_y, color="orange", label="linear trend")
    ax_main.set_title("Linear trend estimation shampoo sales")
    ax_main.legend()

    residuals = sol_line_y - df["Sales"].values
    total_residuals = round(np.abs(residuals).sum(), 1)
    ax_residuals.plot(sol_line_x, residuals, color="red", marker="x", label="residuals")
    ax_residuals.plot(sol_line_x, [0] * df.shape[0], color="grey", alpha=0.5, linestyle="--")
    ax_residuals.set_title(f"Residuals linear trend estimation, sum: {total_residuals}")
    ax_residuals.legend()


def exp_trend(df, ax_main, ax_residuals):
    times = np.arange(df.shape[0]).reshape(df.shape[0], 1)
    X = np.column_stack((np.ones((df.shape[0], 1)), times, np.square(times)))
    y = df["Sales"].values.reshape(df.shape[0], 1)

    pseudo_inv = np.dot(np.linalg.inv(np.dot(X.T, X)), X.T)
    solution = np.dot(pseudo_inv, y)
    x0, x1, x2 = solution[0, 0], solution[1, 0], solution[2, 0]

    sol_line_x = np.arange(df.shape[0])
    sol_line_y = x0 + x1 * sol_line_x + x2 * np.square(sol_line_x)

    ax_main.plot(sol_line_x, df["Sales"].values, color="blue", label="sales")
    ax_main.plot(sol_line_x, sol_line_y, color="orange", label="exp trend")
    ax_main.set_title("Exponential trend estimation shampoo sales")
    ax_main.legend()

    residuals = sol_line_y - df["Sales"].values
    total_residuals = round(np.abs(residuals).sum(), 1)
    ax_residuals.plot(sol_line_x, residuals, color="red", marker="x", label="residuals")
    ax_residuals.plot(sol_line_x, [0] * df.shape[0], color="grey", alpha=0.5, linestyle="--")
    ax_residuals.set_title(f"Residuals exponential trend estimation, sum: {total_residuals}")
    ax_residuals.legend()




def main():
    fig, axes = plt.subplots(2, 2, sharex=False, sharey=False)
    df = pd.read_csv("data/shampoo.csv", index_col="Month")
    linear_trend(df, axes[0, 0], axes[1, 0])
    exp_trend(df, axes[0, 1], axes[1, 1])
    plt.show()

if __name__ == "__main__":
    main()
