"""Evaluate frozen source states using a sealed prospective protocol."""
import argparse

import torch

from profilefield.experiments.external import run_external

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    run_external(args.protocol)
