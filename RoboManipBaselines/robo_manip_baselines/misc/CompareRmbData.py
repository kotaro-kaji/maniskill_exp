import argparse

import numpy as np

from robo_manip_baselines.common import RmbData


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("path1", type=str, help="first path (*.hdf5 or *.rmb)")
    parser.add_argument("path2", type=str, help="second path (*.hdf5 or *.rmb)")

    return parser.parse_args()


class CompareRmbData:
    def __init__(self, path1, path2):
        self.path1 = path1
        self.path2 = path2

    def run(self):
        with (
            RmbData(self.path1) as rmb_data1,
            RmbData(self.path2) as rmb_data2,
        ):
            # Check data
            keys1 = set(rmb_data1.keys())
            keys2 = set(rmb_data2.keys())

            only_in_file1 = keys1 - keys2
            only_in_file2 = keys2 - keys1

            if only_in_file1:
                print(
                    f"[{self.__class__.__name__}] Keys only in {self.path1}: {only_in_file1}"
                )
            if only_in_file2:
                print(
                    f"[{self.__class__.__name__}] Keys only in {self.path2}: {only_in_file2}"
                )

            common_keys = sorted(keys1 & keys2)
            for key in common_keys:
                data1 = rmb_data1[key]
                data2 = rmb_data2[key]

                if data1.shape != data2.shape:
                    print(
                        f"[{self.__class__.__name__}] {key:25}: Shape mismatch. {data1.shape} vs {data2.shape}"
                    )
                    continue

                if data1.dtype != data2.dtype:
                    print(
                        f"[{self.__class__.__name__}] {key:25}: Type mismatch. {data1.dtype} vs {data2.dtype}"
                    )
                    if not (
                        np.issubdtype(data1.dtype, np.floating)
                        and np.issubdtype(data2.dtype, np.floating)
                    ):
                        continue

                if np.issubdtype(data1.dtype, np.unsignedinteger):
                    diff = np.abs(data1[:].astype(np.int64) - data2[:].astype(np.int64))
                else:
                    diff = np.abs(data1[:] - data2[:])
                if np.max(diff) > 0.0:
                    print(
                        f"[{self.__class__.__name__}] {key:25}: Mean err = {np.mean(diff):.5g}, Max err = {np.max(diff):.5g}"
                    )

            # Check attrs
            keys1 = set(rmb_data1.attrs.keys())
            keys2 = set(rmb_data2.attrs.keys())

            only_in_file1 = keys1 - keys2
            only_in_file2 = keys2 - keys1

            if only_in_file1:
                print(
                    f"[{self.__class__.__name__}] Attributes only in {self.path1}: {only_in_file1}"
                )
            if only_in_file2:
                print(
                    f"[{self.__class__.__name__}] Attributes only in {self.path2}: {only_in_file2}"
                )

            common_keys = sorted(keys1 & keys2)
            for key in common_keys:
                data1 = rmb_data1.attrs[key]
                data2 = rmb_data2.attrs[key]

                if type(data1) is not type(data2):
                    print(
                        f"[{self.__class__.__name__}] Attribute {key:25}: Type mismatch. {type(data1)} vs {type(data2)}"
                    )
                    continue

                if isinstance(data1, str):
                    if data1 != data2:
                        print(
                            f"[{self.__class__.__name__}] Attribute {key:25}: Data mismatch: {data1} vs {data2}"
                        )


if __name__ == "__main__":
    compare = CompareRmbData(**vars(parse_argument()))
    compare.run()
