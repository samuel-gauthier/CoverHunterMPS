#!/usr/bin/env python3
# author:liufeng
# datetime:2022/7/11 3:20 PM
# software: PyCharm

import argparse
import logging
import os
import random
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor

import librosa
import numpy as np
import torch

from src.cqt import PyCqt
from src.dataset import SignalAug
from src.utils import (
    RARE_DELIMITER,
    dict_to_line,
    line_to_dict,
    load_hparams,
    read_lines,
    remake_path_for_linux,
    write_lines,
)


def _sort_lines_by_utt(init_path, sorted_path):
    dump_lines = read_lines(init_path, log=False)
    dump_lines = sorted(dump_lines, key=lambda x: (line_to_dict(x)["utt"]))
    write_lines(sorted_path, dump_lines, log=True)


def _remove_dup_line(init_path, new_path):
    logging.info("Remove line with same utt")
    old_line_num = len(read_lines(init_path, log=False))
    utt_set = set()
    valid_lines = []
    for line in read_lines(init_path, log=False):
        utt = line_to_dict(line)["utt"]
        if utt not in utt_set:
            utt_set.add(utt)
            valid_lines.append(line)
    logging.info(f"Filter stage: {old_line_num}->{len(valid_lines)}")
    write_lines(new_path, valid_lines)


def _remove_invalid_line(init_path, new_path):
    old_line_num = len(read_lines(init_path, log=False))
    dump_lines = []
    for line in read_lines(init_path, log=False):
        local_data = line_to_dict(line)
        if not os.path.exists(local_data["wav"]):
            logging.info(f"Unvalid data for wav path: {line}")
            continue
        dump_lines.append(line)
    logging.info(f"Filter stage: {old_line_num}->{len(dump_lines)}")
    write_lines(new_path, dump_lines)


def _remove_line_with_same_dur(init_path, new_path):
    """remove line with same song-id and same dur-ms"""
    old_line_num = len(read_lines(init_path, log=False))
    dump_lines = []
    for line in read_lines(init_path, log=False):
        local_data = line_to_dict(line)
        if not os.path.exists(local_data["wav"]):
            logging.info(f"Unvalid data for wav path: {line}")
            continue
        dump_lines.append(line)
    logging.info(f"Filter stage: {old_line_num}->{len(dump_lines)}")
    write_lines(new_path, dump_lines)


def sox_change_speed(inp_path, out_path, k):
    cmd = "sox -q {} -t wav  -r 16000 -c 1 {} tempo {} " "> sox.log 2> sox.log".format(
        remake_path_for_linux(inp_path), remake_path_for_linux(out_path), k
    )

    try:
        subprocess.call(cmd, shell=True)
        success = os.path.exists(out_path)
        if not success:
            logging.info(f"Error for sox: {cmd}")
        return success
    except RuntimeError:
        logging.info(f"RuntimeError: {cmd}")
        return False
    except EOFError:
        logging.info(f"EOFError: {cmd}")
        return False


# instead of original serial function
# leverage multiple CPU cores to run multiple sox instances in parallel
def _speed_aug_worker(args):
    """worker function for _speed_aug_parallel"""
    line, speed, sp_dir = args
    wav_path = line["wav"]

    if abs(speed - 1.0) > 0.01:
        sp_utt = "sp_{}-{}".format(speed, line["utt"])
        sp_wav_path = os.path.join(sp_dir, f"{sp_utt}.wav")
        if not os.path.exists(sp_wav_path):
            sox_change_speed(wav_path, sp_wav_path, speed)
    else:
        sp_utt = line["utt"]
        sp_wav_path = line["wav"]

    # added logic missing in original CoverHunter: modify dur_s
    # so that _cut_one_line_with_dur function slices augmented samples appropriately
    # since speed augmentation also changes duration
    line["dur_s"] = round(line["dur_s"] / speed, 2)
    line["utt"] = sp_utt
    line["wav"] = sp_wav_path
    return line


def _speed_aug_parallel(init_path, aug_speed_lst, aug_path, sp_dir):
    """add items with speed argument wav"""
    logging.info(f"speed factor: {aug_speed_lst}")
    os.makedirs(sp_dir, exist_ok=True)
    total_lines = read_lines(init_path, log=False)
    dump_lines = []

    with ProcessPoolExecutor() as executor:
        worker_args = [
            (line_to_dict(line), speed, sp_dir)
            for line in total_lines
            for speed in aug_speed_lst
        ]

        for result in executor.map(_speed_aug_worker, worker_args):
            if result == "skip":
                continue
            dump_lines.append(dict_to_line(result))
            if len(dump_lines) % 1000 == 0:
                logging.info(f"{len(dump_lines)}: {dump_lines[-1]}")

    write_lines(aug_path, dump_lines)


# instead of original serial function,
# leverage multiple CPU cores to run multiple CQT extractions in parallel
def _extract_cqt_worker(args):
    """worker function for _extract_cqt_parallel"""
    line, cqt_dir = args
    wav_path = line["wav"]
    py_cqt = PyCqt(sample_rate=16000, hop_size=0.04)
    feat_path = os.path.join(cqt_dir, "{}.cqt.npy".format(line["utt"]))

    if not os.path.exists(feat_path):
        y, sr = librosa.load(wav_path, sr=16000)  # y is a npy ndarray
        y = y / max(0.001, np.max(np.abs(y))) * 0.999
        cqt = py_cqt.compute_cqt(signal_float=y, feat_dim_first=False)
        np.save(feat_path, cqt)
        feat_len = len(cqt)
    else:
        feat_len = len(np.load(feat_path))
    line["feat"] = feat_path
    line["feat_len"] = feat_len
    return line


def _extract_cqt_parallel(init_path, out_path, cqt_dir):
    logging.info("Extract CQT features")
    os.makedirs(cqt_dir, exist_ok=True)
    dump_lines = []

    with ProcessPoolExecutor() as executor:
        worker_args = [
            (line_to_dict(line), cqt_dir) for line in read_lines(init_path, log=False)
        ]

        for result in executor.map(_extract_cqt_worker, worker_args):
            dump_lines.append(dict_to_line(result))
            if len(dump_lines) % 1000 == 0:
                logging.info(
                    "Extracted CQT for {} items: {}".format(
                        len(dump_lines), result["utt"],
                    ),
                )

    write_lines(out_path, dump_lines)


### experimental Torch-optimized MPS use for CQT ###
### not usable as of 21 Feb 2024 ###
def _extract_cqt_workerMPS(args):
    """worker function for _extract_cqt_parallel"""
    line, cqt_dir, device = args
    wav_path = line["wav"]
    py_cqt = PyCqt(sample_rate=16000, hop_size=0.04, mps=True)
    feat_path = os.path.join(cqt_dir, "{}.cqt.npy".format(line["utt"]))

    if not os.path.exists(feat_path):
        y, sr = librosa.load(wav_path, sr=16000)  # y is a npy ndarray
        y = torch.from_numpy(y).to(device)
        y = y / max(0.001, torch.max(torch.abs(y))) * 0.999
        # compute_cqtMPS returns a Torch tensor on 'mps'
        cqt = py_cqt.compute_cqtMPS(signal_float=y, feat_dim_first=False)
        np.save(feat_path, cqt.cpu().numpy())
        feat_len = len(cqt)
    else:
        feat_len = len(np.load(feat_path))
    line["feat"] = feat_path
    line["feat_len"] = feat_len
    return line


### experimental Torch-optimized MPS use for CQT ###
### not usable as of 21 Feb 2024 ###
def _extract_cqt_parallelMPS(init_path, out_path, cqt_dir):
    logging.info("Extract CQT features")
    assert (
        torch.backends.mps.is_available()
    ), "This implementation only runs on Apple M-series chips."
    device = torch.device("mps")
    os.makedirs(cqt_dir, exist_ok=True)
    dump_lines = []
    with ProcessPoolExecutor() as executor:
        worker_args = [
            (line_to_dict(line), cqt_dir, device)
            for line in read_lines(init_path, log=False)
        ]

        for result in executor.map(_extract_cqt_workerMPS, worker_args):
            dump_lines.append(dict_to_line(result))
            if len(dump_lines) % 1000 == 0:
                logging.info(
                    "Extracted CQT for {} items: {}".format(
                        len(dump_lines), result["utt"],
                    ),
                )

    write_lines(out_path, dump_lines)


def _extract_cqt_with_noise(init_path, full_path, cqt_dir, hp_noise):
    logging.info("Extract Cqt feature with noise argumentation")
    os.makedirs(cqt_dir, exist_ok=True)

    py_cqt = PyCqt(sample_rate=16000, hop_size=0.04)
    sig_aug = SignalAug(hp_noise)
    vol_lst = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    dump_lines = []
    for line in read_lines(init_path, log=False):
        local_data = line_to_dict(line)
        wav_path = local_data["wav"]
        local_data["utt"] = local_data["utt"] + "{}noise_{}".format(
            RARE_DELIMITER, hp_noise["name"],
        )
        local_data["feat"] = os.path.join(
            cqt_dir, "{}.cqt.npy".format(local_data["utt"]),
        )

        vol = random.choice(vol_lst)
        if not os.path.exists(local_data["feat"]):
            y, sr = librosa.load(wav_path, sr=16000)
            y = sig_aug.augmentation(y)
            y = y / max(0.001, np.max(np.abs(y))) * 0.999 * vol
            cqt = py_cqt.compute_cqt(signal_float=y, feat_dim_first=False)
            np.save(local_data["feat"], cqt)
            local_data["feat_len"] = len(cqt)

        if "feat_len" not in local_data.keys():
            cqt = np.load(local_data["feat"])
            local_data["feat_len"] = len(cqt)

        dump_lines.append(dict_to_line(local_data))

        if len(dump_lines) % 1000 == 0:
            logging.info(
                "Process cqt for {}items: {}, vol:{}".format(
                    len(dump_lines), local_data["utt"], vol,
                ),
            )

    write_lines(full_path, dump_lines)


def _add_song_id(init_path, out_path, map_path=None):
    """map format:: song_name->song_id"""
    song_id_map = {}
    dump_lines = []
    for line in read_lines(init_path, log=False):
        local_data = line_to_dict(line)
        song_name = local_data["song"]
        if song_name not in song_id_map:
            song_id_map[song_name] = len(song_id_map)
        local_data["song_id"] = song_id_map[song_name]
        dump_lines.append(dict_to_line(local_data))
    write_lines(out_path, dump_lines)

    if map_path:
        dump_lines = []
        for k, v in song_id_map.items():
            dump_lines.append(f"{k} {v}")
        write_lines(map_path, dump_lines)


def _split_data_by_song_id(
    input_path, train_path, val_path, test_path, test_song_ids_path, hp,
):
    """
    Splits data into train and test sets based on song IDs using stratified sampling.

    Args:
        input_path: Path to the input data file.
        train_path: Path to write the training data.
        val_path: Path to write the validation data.
        test_path: Path to write the testing data.
        test_song_ids_path: Path to write a list of song_ids reserved for test
    """
    # percent of unique song IDs to include only in the val or test sets
    val_only_percent = hp["train-sample_unseen"]
    test_only_percent = hp["test_data_unseen"]
    val_ratio = hp["train-sample_data_split"]
    test_ratio = hp["test_data_split"]

    # Dictionary to store song ID counts and shuffled sample lists
    song_data = {}
    for line in read_lines(input_path):
        local_data = line_to_dict(line)
        song_id = local_data["song_id"]
        if song_id not in song_data:
            song_data[song_id] = []
        song_data[song_id].append(local_data)
    logging.info("Number of distinct songs: %s", len(song_data))

    # Separate songs for test-only and stratified split
    num_songs = len(song_data)
    # ensure minimum of one if non-zero intended
    test_only_count = (
        max(1, int(num_songs * test_only_percent)) if test_only_percent > 0 else 0
    )
    # Randomly select songs for test only
    test_only_songs = random.sample(list(song_data.keys()), test_only_count)
    remaining_songs = {
        song_id: samples
        for song_id, samples in song_data.items()
        if song_id not in test_only_songs
    }

    # Process songs for test only (all samples to test)
    test_data = []
    for song_id in test_only_songs:
        test_data.extend(song_data[song_id])

    # Separate songs for val-only and stratified split
    # ensure minimum of one if non-zero intended
    val_only_count = (
        max(1, int(num_songs * val_only_percent)) if val_only_percent > 0 else 0
    )
    val_only_songs = random.sample(
        list(remaining_songs.keys()), val_only_count,
    )  # Randomly select songs for val only
    remaining_songs = {
        song_id: samples
        for song_id, samples in remaining_songs.items()
        if song_id not in val_only_songs
    }

    # Process songs for val only (all samples to val)
    val_data = []
    for song_id in val_only_songs:
        val_data.extend(song_data[song_id])

    train_data, remaining_val_data, remaining_test_data = [], [], []
    if (
        val_ratio > 0 or test_ratio > 0
    ):  # don't bother if 0,0 like for testset CQT generation
        # Stratified split for remaining songs
        for song_id, samples in remaining_songs.items():
            # Randomly shuffle samples for this song ID
            random.shuffle(samples)

            # Calculate val split points based on train ratio and minimum samples (1)
            min_samples = 1  # Ensure at least 1 sample in each set for remaining songs
            val_split = int(len(samples) * val_ratio)
            val_split = max(
                min_samples, val_split,
            )  # Ensure at least min_samples in val

            # Calculate test split points based on train ratio and minimum samples (1)
            min_samples = 1  # Ensure at least 1 sample in each set for remaining songs
            test_split = int(len(samples) * test_ratio)
            test_split = max(
                min_samples, test_split,
            )  # Ensure at least min_samples in test

            remaining_val_data.extend(samples[:val_split])
            remaining_test_data.extend(samples[val_split : val_split + test_split])
            train_data.extend(samples[val_split + test_split :])

        val_data.extend(remaining_val_data)
        test_data.extend(remaining_test_data)
    else:
        train_data.extend(remaining_songs.items())

    logging.info("Number of samples in train: %s", len(train_data))
    logging.info("Number of samples in validate: %s", len(val_data))
    logging.info("Number of samples in test: %s", len(test_data))

    write_lines(train_path, [dict_to_line(sample) for sample in train_data])
    if len(val_data) > 0:
        write_lines(val_path, [dict_to_line(sample) for sample in val_data])
    if len(test_data) > 0:
        write_lines(test_path, [dict_to_line(sample) for sample in test_data])
    if len(test_only_songs) > 0:
        write_lines(
            test_song_ids_path, [dict_to_line(song) for song in test_only_songs],
        )


# =============================================================================
# Not needed
#
# def _add_version_id(init_path, out_path):
#   song_version_map = {}
#   for line in read_lines(init_path, log=False):
#     local_data = line_to_dict(line)
#     song_id = local_data["song_id"]
#     if song_id not in song_version_map.keys():
#       song_version_map[song_id] = []
#     song_version_map[song_id].append(local_data)
#
#   dump_lines = []
#   for k, v_lst in song_version_map.items():
#     for version_id, local_data in enumerate(v_lst):
#       local_data["version_id"] = version_id
#       dump_lines.append(dict_to_line(local_data))
#   write_lines(out_path, dump_lines)
#   return
#
#
# def _extract_song_num(full_path, song_name_map_path, song_id_map_path):
#   """add map of song_id:num and song_name:num"""
#   song_id_num = {}
#   max_song_id = 0
#   for line in read_lines(full_path):
#     local_data = line_to_dict(line)
#     song_id = local_data["song_id"]
#     if song_id not in song_id_num.keys():
#       song_id_num[song_id] = 0
#     song_id_num[song_id] += 1
#     if song_id >= max_song_id:
#       max_song_id = song_id
#   logging.info("max_song_id: {}".format(max_song_id))
#
#   dump_data = list(song_id_num.items())
#   dump_data = sorted(dump_data)
#   dump_lines = ["{} {}".format(k, v) for k, v in dump_data]
#   write_lines(song_id_map_path, dump_lines, log=False)
#
#   song_num = {}
#   for line in read_lines(full_path, log=False):
#     local_data = line_to_dict(line)
#     song_id = local_data["song"]
#     if song_id not in song_num.keys():
#       song_num[song_id] = 0
#     song_num[song_id] += 1
#
#   dump_data = list(song_num.items())
#   dump_data = sorted(dump_data)
#   dump_lines = ["{} {}".format(k, v) for k, v in dump_data]
#   write_lines(song_name_map_path, dump_lines, log=False)
#   return
#
#
# def _sort_lines_by_song_id(full_path, sorted_path):
#   dump_lines = read_lines(full_path, log=False)
#   dump_lines = sorted(dump_lines,
#                       key=lambda x: (int(line_to_dict(x)["song_id"]),
#                                      int(line_to_dict(x)["version_id"])))
#   write_lines(sorted_path, dump_lines, log=True)
#   return
#
# =============================================================================


def _clean_lines(full_path, clean_path):
    dump_lines = []
    for line in read_lines(full_path):
        local_data = line_to_dict(line)
        clean_data = {
            "utt": local_data["utt"],
            "song_id": local_data["song_id"],
            "song": local_data["song"],
            "version_id": local_data["version_id"],
        }
        if "feat" in local_data.keys():
            clean_data.update(
                {"feat_len": local_data["feat_len"], "feat": local_data["feat"]},
            )
        else:
            clean_data.update(
                {"dur_ms": local_data["dur_ms"], "wav": local_data["wav"]},
            )
        dump_lines.append(dict_to_line(clean_data))
    write_lines(clean_path, dump_lines)


def _generate_csi_features(hp, feat_dir, start_stage, end_stage):
    data_path = os.path.join(feat_dir, "dataset.txt")
    assert os.path.exists(data_path)

    init_path = os.path.join(feat_dir, "data.init.txt")
    shutil.copy(data_path, init_path)
    if start_stage <= 0 <= end_stage:
        logging.info("Stage 0: data deduping")
        _sort_lines_by_utt(init_path, init_path)
        _remove_dup_line(init_path, init_path)

    # aug_speed_mode is a list like: [0.8, 0.9, 1.0, 1.1, 1.2]
    # do include 1.0 to include original speed.
    # Anything between .99 and 1.01 will be ignored, instead passing along the original file.
    sp_aug_path = os.path.join(feat_dir, "sp_aug.txt")
    if start_stage <= 3 <= end_stage:
        logging.info("Stage 3: speed augmentation")
        if "aug_speed_mode" in hp.keys() and not os.path.exists(sp_aug_path):
            sp_dir = os.path.join(feat_dir, "sp_wav")
            #      _speed_aug(init_path, hp["aug_speed_mode"], sp_aug_path, sp_dir)
            _speed_aug_parallel(init_path, hp["aug_speed_mode"], sp_aug_path, sp_dir)

    new_full = False
    full_path = os.path.join(feat_dir, "full.txt")
    if start_stage <= 4 <= end_stage:
        logging.info("Stage 4: extract cqt feature")
        if not os.path.exists(full_path):
            new_full = True
            cqt_dir = os.path.join(feat_dir, "cqt_feat")
            #    _extract_cqt(sp_aug_path, full_path, cqt_dir)
            #    _extract_cqt_parallelMPS(sp_aug_path, full_path, cqt_dir)
            # Failed attempt to do MPS acceleration of CQT.
            # Too many unsupported demands on Torch MPS implementation as of Feb 2024
            # Got it to run w/o errors but output wasn't quite right,
            # and speed was 7 min 28 s for covers80,
            # compared to 2 min 5 s with CPU-based _extract_cqt_parallel()
            _extract_cqt_parallel(sp_aug_path, full_path, cqt_dir)

    # noise augmentation was default off for CoverHunter
    hp_noise = hp.get("add_noise", None)
    if (
        start_stage <= 5 <= end_stage
        and hp_noise
        and os.path.exists(hp_noise["noise_path"])
    ):
        logging.info("Stage 5: add noise and extract cqt feature")
        noise_cqt_dir = os.path.join(feat_dir, "cqt_with_noise")
        _extract_cqt_with_noise(
            full_path, full_path, noise_cqt_dir, hp_noise={"add_noise": hp_noise},
        )

    # assumes song titles provided in dataset.txt are unique identifiers for the parent songs
    if start_stage <= 8 <= end_stage:
        logging.info("Stage 8: add song_id")
        #    song_id_map_path = os.path.join(feat_dir, "song_id.map")
        if new_full or not os.path.exists(full_path):
            _add_song_id(full_path, full_path)

    # =============================================================================
    # CoverHunter doesn't actually do anything with version_id or the .map files
    # if start_stage <= 9 <= end_stage:
    #  logging.info("Stage 9: add version_id")
    #   _add_version_id(full_path, full_path)
    #
    # if start_stage <= 10 <= end_stage:
    #   logging.info("Start stage 10: extract version num")
    #   song_id_map_path = os.path.join(feat_dir, "song_id_num.map")
    #   song_num_map_path = os.path.join(feat_dir, "song_name_num.map")
    #   _extract_song_num(full_path, song_num_map_path, song_id_map_path)
    #
    # if start_stage <= 11 <= end_stage:
    #   logging.info("Stage 11: clean for unused keys")
    #   _sort_lines_by_song_id(full_path, full_path)
    # =============================================================================

    if start_stage <= 13 <= end_stage:
        logging.info("Stage 13:Split data into train / validate / test sets")
        train_path = os.path.join(feat_dir, "train.txt")
        val_path = os.path.join(feat_dir, "train-sample.txt")
        test_path = os.path.join(feat_dir, "dev.txt")
        test_song_ids_path = os.path.join(feat_dir, "dev-only-song-ids.txt")
        _split_data_by_song_id(
            full_path, train_path, val_path, test_path, test_song_ids_path, hp,
        )


def _cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument("feat_dir", help="feat_dir")
    parser.add_argument("--start_stage", type=int, default=0)
    parser.add_argument("--end_stage", type=int, default=100)
    args = parser.parse_args()
    hp_path = os.path.join(args.feat_dir, "hparams.yaml")
    hp = load_hparams(hp_path)
    print(hp)
    _generate_csi_features(hp, args.feat_dir, args.start_stage, args.end_stage)


if __name__ == "__main__":
    _cmd()
