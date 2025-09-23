import argparse

import numpy as np

from robo_manip_baselines.common import (
    DataKey,
    RmbData,
    get_pose_from_se3,
    get_se3_from_pose,
    get_se3_from_rel_pose,
    get_skipped_data_seq,
)


def add_arrays_to_smaller_length(arr1, arr2):
    min_length = min(arr1.shape[0], arr2.shape[0])
    return arr1[:min_length] + arr2[:min_length]


def test_get_skipped_data_seq_joint_pos(filename, skip=3):
    abs_key = DataKey.MEASURED_JOINT_POS
    rel_key = DataKey.get_rel_key(abs_key)

    with RmbData(filename) as rmb_data:
        joint_abs_seq = rmb_data[abs_key][:]
        joint_rel_seq = rmb_data[rel_key][:]

    skipped_joint_abs_seq = joint_abs_seq[::skip][1:]

    skipped_joint_rel_seq = get_skipped_data_seq(joint_rel_seq[1:], rel_key, skip)

    skipped_joint_abs_seq2 = add_arrays_to_smaller_length(
        joint_abs_seq[::skip], skipped_joint_rel_seq
    )

    error = np.sum(
        np.abs(
            add_arrays_to_smaller_length(
                skipped_joint_abs_seq, -1 * skipped_joint_abs_seq2
            )
        )
    )

    print(f"[test_get_skipped_data_seq_joint_pos] error: {error}")


def test_get_skipped_data_seq_eef_pose(filename, skip=3):
    abs_key = DataKey.MEASURED_EEF_POSE
    rel_key = DataKey.get_rel_key(abs_key)

    with RmbData(filename) as rmb_data:
        eef_abs_seq = rmb_data[abs_key][:]
        eef_rel_seq = rmb_data[rel_key][:]

    skipped_eef_abs_seq = eef_abs_seq[::skip][1:]

    skipped_eef_rel_seq = get_skipped_data_seq(eef_rel_seq[1:], rel_key, skip)

    skipped_eef_abs_seq2 = []
    for eef_abs, eef_rel in zip(eef_abs_seq[::skip][:-1], skipped_eef_rel_seq):
        skipped_eef_abs_seq2.append(
            get_pose_from_se3(
                get_se3_from_pose(eef_abs) * get_se3_from_rel_pose(eef_rel)
            )
        )
    skipped_eef_abs_seq2 = np.array(skipped_eef_abs_seq2)

    error = np.sum(
        np.abs(
            add_arrays_to_smaller_length(skipped_eef_abs_seq, -1 * skipped_eef_abs_seq2)
        )
    )

    print(f"[test_get_skipped_data_seq_eef_pose] error: {error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "filename",
        type=str,
        help="filename of teleoperation data",
    )
    args = parser.parse_args()

    test_get_skipped_data_seq_joint_pos(args.filename)
    test_get_skipped_data_seq_eef_pose(args.filename)
