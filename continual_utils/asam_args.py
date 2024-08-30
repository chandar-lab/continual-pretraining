import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoothing", default=0.1, type=float, help="Label smoothing.")
    args = parser.parse_args()