import os
import sys
import argparse
import torch
import soundfile as sf
import yaml
import logging
import numpy as np
import time
import glob

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # NOQA: E402
sys.path.insert(0, os.path.dirname(ROOT_PATH))  # NOQA: E402

try:
    from kantts.utils.log import logging_to_file
except ImportError:
    raise ImportError("Please install kantts.")

logging.basicConfig(
    #  filename=os.path.join(stage_dir, 'stdout.log'),
    format="%(asctime)s, %(levelname)-4s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model(ckpt, config=None):
    # load config if not provided
    if config is None:
        dirname = os.path.dirname(os.path.dirname(ckpt))
        config = os.path.join(dirname, "config.yaml")
        with open(config) as f:
            config = yaml.load(f, Loader=yaml.Loader)

    # lazy load for circular error
    from kantts.models.hifigan.hifigan import Generator

    model = Generator(**config["Model"]["Generator"]["params"])
    states = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(states["model"]["generator"])

    # add pqmf if needed
    if config["Model"]["Generator"]["params"]["out_channels"] > 1:
        # lazy load for circular error
        from kantts.models.pqmf import PQMF

        model.pqmf = PQMF()

    return model


def hifigan_infer(input_mel, ckpt_path, output_dir, config=None):
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda", 0)

    if config is not None:
        with open(config, "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)
    else:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(ckpt_path)), "config.yaml"
        )
        if not os.path.exists(config_path):
            raise ValueError("config file not found: {}".format(config_path))
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)

    for key, value in config.items():
        logging.info(f"{key} = {value}")

    # check directory existence
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logging_to_file(os.path.join(output_dir, "stdout.log"))

    if os.path.isfile(input_mel):
        mel_lst = [input_mel]
    elif os.path.isdir(input_mel):
        mel_lst = glob.glob(os.path.join(input_mel, "*mel.npy"))
    else:
        raise ValueError("input_mel should be a file or a directory")

    model = load_model(ckpt_path, config)

    logging.info(f"Loaded model parameters from {ckpt_path}.")
    model.remove_weight_norm()
    model = model.eval().to(device)

    with torch.no_grad():
        start = time.time()
        pcm_len = 0
        for mel in mel_lst:
            utt_id = os.path.splitext(os.path.basename(mel))[0]
            mel_data = np.load(mel)
            # generate
            mel_data = torch.tensor(mel_data, dtype=torch.float).to(device)
            # (T, C) -> (B, C, T)
            mel_data = mel_data.transpose(1, 0).unsqueeze(0)
            y = model(mel_data)
            if hasattr(model, "pqmf"):
                y = model.pqmf.synthesis(y)
            y = y.view(-1).cpu().numpy()
            pcm_len += len(y)

            # save as PCM 16 bit wav file
            sf.write(
                os.path.join(output_dir, f"{utt_id}_gen.wav"),
                y,
                config["audio_config"]["sampling_rate"],
                "PCM_16",
            )
        rtf = (time.time() - start) / (
            pcm_len / config["audio_config"]["sampling_rate"]
        )

    # report average RTF
    logging.info(
        f"Finished generation of {len(mel_lst)} utterances (RTF = {rtf:.03f})."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer hifigan model")
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--input_mel",
        type=str,
        required=True,
        help="Path to input mel file or directory containing mel files",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to output directory"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    args = parser.parse_args()
    hifigan_infer(
        args.input_mel,
        args.ckpt,
        args.output_dir,
        args.config,
    )
