#!/usr/bin/env python3

import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.dataset import AudioFeatDataset, MPerClassSampler
from src.eval_testset import eval_for_map_with_feat
from src.model import Model
from src.scheduler import UserDefineExponentialLR
from src.trainer import load_checkpoint, save_checkpoint, train_one_epoch, validate
from src.utils import create_logger, get_hparams_as_string, load_hparams


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Train: python3 -m tools.train model_dir",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("model_dir")
    parser.add_argument(
        "--first_eval",
        default=False,
        action="store_true",
        help="Set for run eval first before train",
    )
    parser.add_argument(
        "--only_eval",
        default=False,
        action="store_true",
        help="Set for run eval first before train",
    )
    parser.add_argument(
        "--debug", default=False, action="store_true", help="give more debug log",
    )
    parser.add_argument(
        "--runid",
        default="",
        action="store",
        help="put TensorBoard logs in this subfolder of ../logs/",
    )
    args = parser.parse_args()
    model_dir = args.model_dir
    first_eval = args.first_eval
    only_eval = args.only_eval
    run_id = args.runid
    first_eval = True if only_eval else first_eval

    logger = create_logger()
    hp = load_hparams(os.path.join(model_dir, "config/hparams.yaml"))
    match hp["device"]:  # noqa requires python 3.10
        case "mps":
            if not torch.backends.mps.is_available():
                logger.error(
                    "You requested 'mps' device in your hyperparameters"
                    "but you are not running on an Apple M-series chip or "
                    "have not compiled PyTorch for MPS support."
                )
                sys.exit()
            device = torch.device("mps")
        case "cuda":
            if not torch.cuda.is_available():
                logger.error(
                    "You requested 'cuda' device in your hyperparameters"
                    "but you do not have a CUDA-compatible GPU available."
                )
                sys.exit()
            device = torch.device("cuda")
        case _:
            logger.error(
                "You set device: %s"
                " in your hyperparameters but that is not a valid option or is an untested option.",
                 hp["device"]
            )
            sys.exit()

    logger.info("%s", get_hparams_as_string(hp))

    # Initialize variables for early stopping
    best_validation_loss = float("inf")
    early_stopping_counter = 0
    early_stopping_patience = hp["early_stopping_patience"]

    torch.manual_seed(hp["seed"])

    # We use multi length sample to train
    train_loader_lst = []
    for chunk_len in hp["chunk_frame"]:
        train_dataset = AudioFeatDataset(
            hp,
            hp["train_path"],
            train=True,
            mode=hp["mode"],
            chunk_len=chunk_len * hp["mean_size"],
            logger=logger,
        )
        sampler = MPerClassSampler(
            data_path=hp["train_path"],
            m=hp["m_per_class"],
            batch_size=hp["batch_size"],
            distribute=False,
            logger=logger,
        )
        train_loader = DataLoader(
            train_dataset,
            num_workers=hp["num_workers"],
            shuffle=(sampler is None),
            sampler=sampler,
            batch_size=hp["batch_size"],
            pin_memory=True,
            drop_last=True,
        )
        train_loader_lst.append(train_loader)

    # At inference stage, we only use chunk with fixed length
    logger.info("Init train-sample and dev data loader")
    infer_len = hp["chunk_frame"][0] * hp["mean_size"]
    if "train_sample_path" in hp:
        # hp["batch_size"] = 1
        dataset = AudioFeatDataset(
            hp,
            hp["train_sample_path"],
            train=False,
            chunk_len=infer_len,
            mode=hp["mode"],
            logger=logger,
        )
        sampler = MPerClassSampler(
            data_path=hp["train_sample_path"],
            # m=hp["m_per_class"],
            m=1,
            batch_size=hp["batch_size"],
            distribute=False,
            logger=logger,
        )
        train_sampler_loader = DataLoader(
            dataset,
            num_workers=1,
            shuffle=False,
            sampler=sampler,
            batch_size=hp["batch_size"],
            pin_memory=True,
            collate_fn=None,
            drop_last=False,
        )
    else:
        train_sampler_loader = None

    if "dev_path" in hp:
        dataset = AudioFeatDataset(
            hp, hp["dev_path"], chunk_len=infer_len, mode=hp["mode"], logger=(logger),
        )
        sampler = MPerClassSampler(
            data_path=hp["dev_path"],
            m=hp["m_per_class"],
            batch_size=hp["batch_size"],
            distribute=False,
            logger=logger,
        )
        dev_loader = DataLoader(
            dataset,
            num_workers=1,
            shuffle=False,
            sampler=sampler,
            batch_size=hp["batch_size"],
            pin_memory=True,
            collate_fn=None,
            drop_last=False,
        )
    else:
        dev_loader = None

    # we use map-reduce mode to update model when its parameters changed
    # (model.join), that means we do not need to wait one step of all gpu to
    # complete. Pytorch distribution support variable trained samples of different
    # gpus.
    # And also, we compute train-sample/dev/testset on different gpu within epoch.
    # For example: we compute dev at rank0 when epoch 1, when dev is computing,
    # rank1 is going on training and update parameters. When epoch 2, we change
    # to compute dev at rank1, to make sure all ranks run the same train steps
    # almost.

    # test_set_list stores whichever members of all_test_set_list are listed in hparams.yaml
    # default CoverHunter only included "covers80"
    all_test_set_list = ["covers80", "shs_test", "dacaos", "hymf_20", "hymf_100"]
    test_set_list = [d for d in all_test_set_list if d in hp]

    model = Model(hp).to(device)
    checkpoint_dir = os.path.join(model_dir, "pt_model")
    os.makedirs(checkpoint_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), hp["learning_rate"], betas=[hp["adam_b1"], hp["adam_b2"]],
    )
    step, init_epoch = load_checkpoint(model, optimizer, checkpoint_dir, advanced=False)

    scheduler = UserDefineExponentialLR(
        optimizer, gamma=hp["lr_decay"], min_lr=hp["min_lr"], last_epoch=init_epoch,
    )

    log_path = os.path.join(model_dir, "logs", run_id)
    os.makedirs(log_path, exist_ok=True)
    sw = SummaryWriter(log_path)
    if only_eval:
        sw = None

    for epoch in range(max(0, 1 + init_epoch), 100000):
        if not first_eval:
            start = time.time()
            train_step = None
            logger.info("Start to train for epoch %d", epoch)
            step = train_one_epoch(
                model,
                optimizer,
                scheduler,
                train_loader_lst,
                step,
                train_step=train_step,
                device=device,
                sw=sw,
                logger=logger,
            )
            if epoch % hp["every_n_epoch_to_save"] == 0:
                save_checkpoint(model, optimizer, step, epoch, checkpoint_dir)
            logger.info(
                "Time for train epoch %d step %d is %.1fs\n", epoch, step, time.time() - start
            )

        if train_sampler_loader and epoch % hp["every_n_epoch_to_dev"] == 0:
            start = time.time()
            logger.info("compute train-sample at epoch-%d", epoch)

            res = validate(
                model,
                train_sampler_loader,
                "train_sample",
                epoch_num=epoch,
                device=device,
                sw=sw,
                logger=logger,
            )
            logger.info(
                "count:%d, avg_ce_loss:%d", res["count"], res["ce_loss"] / res["count"]
            )

            logger.info(
                "Time for train-sample is %.1fs\n", time.time() - start
            )

        if dev_loader and epoch % hp["every_n_epoch_to_dev"] == 0:
            start = time.time()
            logger.info("compute dev at epoch-%d", epoch)
            dev_res = validate(
                model,
                dev_loader,
                "dev",
                epoch_num=epoch,
                device=device,
                sw=sw,
                logger=logger,
            )
            validation_loss = dev_res["ce_loss"] / dev_res["count"]

            logger.info(
                "count:%s, avg_ce_loss:%s", dev_res["count"], validation_loss
            )
            logger.info("Time for dev is %.1fs\n", time.time() - start)

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

        valid_testlist = []
        for testset_name in test_set_list:
            hp_test = hp[testset_name]
            if epoch % hp_test["every_n_epoch_to_dev"] == 0:
                valid_testlist.append(testset_name)

        for testset_name in valid_testlist:
            hp_test = hp[testset_name]
            logger.info("Compute %s at epoch: %s", testset_name, epoch)

            start = time.time()
            save_name = hp_test.get("save_name", testset_name)
            embed_dir = os.path.join(model_dir, f"embed_{epoch}_{save_name}")
            query_in_ref_path = hp_test.get("query_in_ref_path", None)
            mean_ap, hit_rate, _ = eval_for_map_with_feat(
                hp,
                model,
                embed_dir,
                query_path=hp_test["query_path"],
                ref_path=hp_test["ref_path"],
                query_in_ref_path=query_in_ref_path,
                batch_size=hp["batch_size"],
                device=device,
                logger=logger,
            )

            sw.add_scalar(f"mAP/{testset_name}", mean_ap, epoch)
            sw.add_scalar(f"hit_rate/{testset_name}", hit_rate, epoch)
            logger.info(
                "Test %s, hit_rate:%s, map:%s", testset_name, hit_rate, mean_ap
            )
            logger.info(
                "Time for test-%s is %d sec\n", testset_name, int(time.time() - start)
            )

        if early_stopping_counter >= early_stopping_patience:
            logger.info(
                "Early stopping at epoch %d due to lack of avg_ce_loss"
                "(focal aka cross-entropy loss) improvement.", epoch
            )
            break

        if only_eval:
            return
        first_eval = False
    return


if __name__ == "__main__":
    _main()
