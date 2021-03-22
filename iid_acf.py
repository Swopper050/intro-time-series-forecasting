import argparse

from utils import get_iid_noise_series, plot_ACVF


def main(args):
    series = get_iid_noise_series()
    plot_ACVF(series, max_lag=args.max_lag)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-lag", type=int, default=40)
    args = parser.parse_args()
    main(args)
