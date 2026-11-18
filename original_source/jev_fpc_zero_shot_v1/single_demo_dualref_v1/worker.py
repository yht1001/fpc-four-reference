"""Run the existing frozen-model 0.5 s JEV requests on one GPU shard."""

import argparse

from setup_study import ROOT
from infer import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=("support", "query"))
    parser.add_argument("rank", type=int)
    parser.add_argument("world", type=int)
    args = parser.parse_args()
    if args.split == "query" and not (ROOT / "LOCK.json").is_file():
        raise RuntimeError("Support calibration must be frozen before Query inference")
    main(args.split, args.rank, args.world)
