import os
import sys
import argparse
import torch
import soundfile as sf
import yaml
import logging
import numpy as np
import time

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


def hifigan_infer(audio_config, model_config, ckpt_path, input_mel, output_dir):
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda", 0)

    if audio_config is not None and model_config is not None:
        with open(audio_config, "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)

        with open(model_config, "r") as f:
            config.update(yaml.load(f, Loader=yaml.Loader))
    else:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(ckpt_path)), "config.yaml"
        )
        if not os.path.exists(config_path):
            raise ValueError("config file not found: {}".format(config_path))
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)

    logging_to_file(os.path.join(output_dir, "stdout.log"))

    for key, value in config.items():
        logging.info(f"{key} = {value}")

    # check directory existence
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    utt_id = os.path.splitext(os.path.basename(input_mel))[0]
    if input_mel.endswith(".pt"):
        mel_data = torch.load(input_mel).transpose(0, 1).numpy()
    else:
        mel_data = np.load(input_mel)

    model = load_model(ckpt_path, config)

    logging.info(f"Loaded model parameters from {ckpt_path}.")
    model.remove_weight_norm()
    model = model.eval().to(device)

    with torch.no_grad():
        # generate
        mel_data = torch.tensor(mel_data, dtype=torch.float).to(device)
        # (T, C) -> (B, C, T)
        mel_data = mel_data.transpose(1, 0).unsqueeze(0)
        start = time.time()
        y = model(mel_data).view(-1).cpu().numpy()
        rtf = (time.time() - start) / (len(y) / config["audio_config"]["sampling_rate"])

        # save as PCM 16 bit wav file
        sf.write(
            os.path.join(output_dir, f"{utt_id}_gen.wav"),
            y,
            config["audio_config"]["sampling_rate"],
            "PCM_16",
        )

    # report average RTF
    logging.info(f"Finished generation of 1 utterances (RTF = {rtf:.03f}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer hifigan model")
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--input_mel", type=str, required=True, help="Path to input mel file"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to output directory"
    )
    parser.add_argument(
        "--audio_config", type=str, default=None, help="Path to audio config file"
    )
    parser.add_argument(
        "--model_config", type=str, default=None, help="Path to model config file"
    )
    args = parser.parse_args()
    hifigan_infer(
        args.audio_config,
        args.model_config,
        args.ckpt,
        args.input_mel,
        args.output_dir,
    )
