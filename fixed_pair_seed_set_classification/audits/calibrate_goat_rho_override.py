"""Run the same fixed-A calibration at explicitly recorded alternative rhos."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from . import calibrate_goat_high_rho as calibration


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--er-rho", type=float, required=True)
    parser.add_argument("--paper-rho", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    assert 0 <= args.er_rho <= 1 and 0 <= args.paper_rho <= 1
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "er_rho": args.er_rho, "paper_rho": args.paper_rho,
        "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    path = args.output / "rho_overrides.json"
    if path.exists():
        assert json.loads(path.read_text()) == manifest, "Different rho configuration"
    else:
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    calibration.SETTINGS["sparse_er"]["rho"] = args.er_rho
    calibration.SETTINGS["paper_pa"]["rho"] = args.paper_rho
    sys.argv = [sys.argv[0], "--output", str(args.output), *remaining]
    calibration.main()


if __name__ == "__main__":
    main()
