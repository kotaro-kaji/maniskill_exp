import argparse
import os

from robo_manip_baselines.common import RmbData


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("src_path", type=str, help="source path (*.hdf5 or *.rmb)")
    parser.add_argument("dst_path", type=str, help="destination path (*.hdf5 or *.rmb)")
    parser.add_argument("--force_overwrite", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_argument()

    with RmbData(args.src_path) as rmb_data:
        _, dst_ext = os.path.splitext(args.dst_path.rstrip("/"))
        if dst_ext.lower() == ".rmb":
            rmb_data.dump_to_rmb(args.dst_path, args.force_overwrite)
        elif dst_ext.lower() == ".hdf5":
            rmb_data.dump_to_hdf5(args.dst_path, args.force_overwrite)
        else:
            raise ValueError(
                f"[ConvertRmbData] Invalid file extension '{dst_ext}'. Expected '.hdf5' or '.rmb': {args.dst_path}"
            )
