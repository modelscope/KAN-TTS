import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
import logging
import time
import yaml

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # NOQA: E402
sys.path.insert(0, os.path.dirname(ROOT_PATH))  # NOQA: E402

try:
    from kantts.models import model_builder
    from kantts.train.loss import criterion_builder
    from kantts.datasets.dataset import get_voc_datasets
    from kantts.train.trainer import GAN_Trainer, distributed_init
    from kantts.utils.log import logging_to_file, get_git_revision_hash
except ImportError:
    raise ImportError("Please install kantts.")

logging.basicConfig(
    #  filename=os.path.join(stage_dir, 'stdout.log'),
    format="%(asctime)s, %(levelname)-4s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.DEBUG,
)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


#  TODO: distributed training
def train(
    model_config,
    root_dir,
    stage_dir,
    resume_path=None,
    local_rank=0,
):
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        torch.backends.cudnn.benchmark = True
        logging.info("Args local rank: {}".format(local_rank))
        distributed, device, local_rank, world_size = distributed_init()

    if local_rank != 0:
        sys.stdout = open(os.devnull, "w")
        logger = logging.getLogger()
        logger.disabled = True

    # TODO: make sure all root dir audio_configs are the same
    if not isinstance(root_dir, list):
        root_dir = [root_dir]

    if local_rank == 0 and not os.path.exists(stage_dir):
        os.makedirs(stage_dir)

    audio_config = os.path.join(root_dir[0], "audio_config.yaml")
    with open(audio_config, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    with open(model_config, "r") as f:
        config.update(yaml.load(f, Loader=yaml.Loader))

    logging_to_file(os.path.join(stage_dir, "stdout.log"))

    #  TODO: record some info in config, such as create time, git commit revision
    config["create_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    config["git_revision_hash"] = get_git_revision_hash()

    with open(os.path.join(stage_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f, Dumper=yaml.Dumper, default_flow_style=None)

    for key, value in config.items():
        logging.info(f"{key} = {value}")

    if distributed:
        config["rank"] = torch.distributed.get_rank()
        config["distributed"] = True

    #  TODO: abstract dataloader
    # Dataset prepare
    train_dataset, valid_dataset = get_voc_datasets(
        root_dir,
        config["audio_config"]["sampling_rate"],
        config["audio_config"]["n_fft"],
        config["audio_config"]["hop_length"],
        config["allow_cache"],
        config["batch_max_steps"],
    )

    logging.info(f"The number of training files = {len(train_dataset)}.")
    logging.info(f"The number of validation files = {len(valid_dataset)}.")

    sampler = {"train": None, "valid": None}
    if distributed:
        # setup sampler for distributed training
        from torch.utils.data.distributed import DistributedSampler

        sampler["train"] = DistributedSampler(
            dataset=train_dataset,
            num_replicas=world_size,
            shuffle=True,
        )
        sampler["valid"] = DistributedSampler(
            dataset=valid_dataset,
            num_replicas=world_size,
            shuffle=False,
        )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=False if distributed else True,
        collate_fn=train_dataset.collate_fn,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        sampler=sampler["train"],
        pin_memory=config["pin_memory"],
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        shuffle=False if distributed else True,
        collate_fn=valid_dataset.collate_fn,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        sampler=sampler["valid"],
        pin_memory=config["pin_memory"],
    )

    model, optimizer, scheduler = model_builder(config, device, local_rank, distributed)

    criterion = criterion_builder(config, device)

    logging.info(model["generator"])
    logging.info(
        "Generator mdoel parameters count: {}".format(
            count_parameters(model["generator"])
        )
    )
    logging.info(model["discriminator"])
    logging.info(optimizer["generator"])
    logging.info(optimizer["discriminator"])
    logging.info(scheduler["generator"])
    logging.info(scheduler["discriminator"])
    for criterion_ in criterion.values():
        logging.info(criterion_)

    trainer = GAN_Trainer(
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        sampler=sampler,
        train_loader=train_dataloader,
        valid_loader=valid_dataloader,
        max_steps=config["train_max_steps"],
        save_dir=stage_dir,
        save_interval=config["save_interval_steps"],
        valid_interval=config["eval_interval_steps"],
        log_interval=config["log_interval_steps"],
    )

    if resume_path is not None:
        trainer.load_checkpoint(resume_path)
        logging.info(f"Successfully resumed from {args.resume}.")

    try:
        trainer.train()
    except KeyboardInterrupt:
        trainer.save_checkpoint(
            os.path.join(
                os.path.join(stage_dir, "ckpt"), f"checkpoint-{trainer.steps}.pth"
            )
        )
        logging.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model for speech synthesis")

    parser.add_argument(
        "--model_config", type=str, required=True, help="model config file"
    )
    parser.add_argument(
        "--root_dir",
        nargs="+",
        type=str,
        required=True,
        help="root dir of dataset; cloud be multiple directories",
    )
    parser.add_argument(
        "--stage_dir",
        type=str,
        required=True,
        help="stage dir of checkpoint, log and intermidate results ",
    )
    parser.add_argument(
        "--resume_path", type=str, default=None, help="path to resume checkpoint"
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )
    args = parser.parse_args()

    train(
        args.model_config,
        args.root_dir,
        args.stage_dir,
        args.resume_path,
        args.local_rank,
    )
