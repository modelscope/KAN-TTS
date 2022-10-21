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
    from kantts.datasets.dataset import get_bert_text_datasets
    from kantts.train.trainer import distributed_init, Textsy_BERT_Trainer
    from kantts.utils.log import logging_to_file, get_git_revision_hash
except ImportError:
    raise ImportError("Please install kantts.")

logging.basicConfig(
    #  filename=os.path.join(stage_dir, 'stdout.log'),
    format="%(asctime)s %(levelname)-4s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
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
        distributed = False
    else:
        torch.backends.cudnn.benchmark = True
        logging.info("Args local rank: {}".format(local_rank))
        distributed, device, local_rank, world_size = distributed_init()

    if local_rank != 0:
        sys.stdout = open(os.devnull, "w")
        logger = logging.getLogger()
        logger.disabled = True

    if not isinstance(root_dir, list):
        root_dir = [root_dir]

    meta_file = [os.path.join(d, "raw_metafile.txt") for d in root_dir]

    if local_rank == 0 and not os.path.exists(stage_dir):
        os.makedirs(stage_dir)

    with open(model_config, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    logging_to_file(os.path.join(stage_dir, "stdout.log"))

    # TODO: record some info in config, such as create time, git commit
    # revision
    config["create_time"] = time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime())
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
    train_dataset, valid_dataset = get_bert_text_datasets(
        meta_file, root_dir, config, config["allow_cache"]
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

    ling_unit_size = train_dataset.ling_unit.get_unit_size()

    config["Model"]["KanTtsTextsyBERT"]["params"].update(ling_unit_size)
    model, optimizer, scheduler = model_builder(
        config, device, local_rank, distributed)

    criterion = criterion_builder(config, device)

    logging.info(model["KanTtsTextsyBERT"])
    logging.info(
        "TextsyBERT mdoel parameters count: {}".format(
            count_parameters(model["KanTtsTextsyBERT"])
        )
    )

    logging.info(optimizer["KanTtsTextsyBERT"])
    logging.info(scheduler["KanTtsTextsyBERT"])
    for criterion_ in criterion.values():
        logging.info(criterion_)

    trainer = Textsy_BERT_Trainer(
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
        grad_clip=config["grad_norm"],
    )

    if resume_path is not None:
        trainer.load_checkpoint(resume_path, True)
        logging.info(f"Successfully resumed from {resume_path}.")

    try:
        trainer.train()
    except (Exception, KeyboardInterrupt) as e:
        logging.error(e, exc_info=True)
        trainer.save_checkpoint(
            os.path.join(
                os.path.join(
                    stage_dir,
                    "ckpt"),
                f"checkpoint-{trainer.steps}.pth"))
        logging.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model for speech synthesis")

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
        "--resume_path",
        type=str,
        default=None,
        help="path to resume checkpoint")
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="local rank for distributed training")
    args = parser.parse_args()

    train(
        args.model_config,
        args.root_dir,
        args.stage_dir,
        args.resume_path,
        args.local_rank,
    )
