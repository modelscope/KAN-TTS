import os
import sys
import argparse
import yaml
import logging
import zipfile
from glob import glob
import soundfile as sf
import numpy as np


ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # NOQA: E402
sys.path.insert(0, os.path.dirname(ROOT_PATH))  # NOQA: E402

try:
    from kantts.bin.infer_sambert import am_infer
    from kantts.bin.infer_hifigan import hifigan_infer
    from kantts.utils.ling_unit import text_to_mit_symbols as text_to_symbols
except ImportError:
    raise ImportError("Please install kantts.")

logging.basicConfig(
    #  filename=os.path.join(stage_dir, 'stdout.log'),
    format="%(asctime)s, %(levelname)-4s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)


def concat_process(chunked_dir, output_dir):
    wav_files = sorted(glob(os.path.join(chunked_dir, "*.wav")))
    print(wav_files)
    sentence_sil = 0.28  # seconds
    end_sil = 0.05  # seconds

    cnt = 0
    wav_concat = None
    main_id, sub_id = 0, 0

    while cnt < len(wav_files):
        wav_file = os.path.join(
            chunked_dir, "{}_{}_mel_gen.wav".format(main_id, sub_id)
        )
        if os.path.exists(wav_file):
            wav, sr = sf.read(wav_file)
            sentence_sil_samples = int(sentence_sil * sr)
            end_sil_samples = int(end_sil * sr)
            if sub_id == 0:
                wav_concat = wav
            else:
                wav_concat = np.concatenate(
                    (wav_concat, np.zeros(sentence_sil_samples), wav), axis=0
                )

            sub_id += 1
            cnt += 1
        else:
            if wav_concat is not None:
                wav_concat = np.concatenate(
                    (wav_concat, np.zeros(end_sil_samples)), axis=0
                )
                sf.write(os.path.join(output_dir, f"{main_id}.wav"), wav_concat, sr)

            main_id += 1
            sub_id = 0
            wav_concat = None

        if cnt == len(wav_files):
            wav_concat = np.concatenate((wav_concat, np.zeros(end_sil_samples)), axis=0)
            sf.write(os.path.join(output_dir, f"{main_id}.wav"), wav_concat, sr)


def text_to_wav(
    text_file,
    output_dir,
    resources_zip_file,
    am_ckpt,
    voc_ckpt,
    speaker=None,
    lang="PinYin",
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "res_wavs"), exist_ok=True)

    resource_root_dir = os.path.dirname(resources_zip_file)
    resource_dir = os.path.join(resource_root_dir, "resource")

    if not os.path.exists(resource_dir):
        logging.info("Extracting resources...")
        with zipfile.ZipFile(resources_zip_file, "r") as zip_ref:
            zip_ref.extractall(resource_root_dir)

    with open(text_file, "r") as text_data:
        texts = text_data.readlines()

    logging.info("Converting text to symbols...")
    am_config = os.path.join(os.path.dirname(os.path.dirname(am_ckpt)), "config.yaml")
    with open(am_config, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    if speaker is None:
        speaker = config["linguistic_unit"]["speaker_list"].split(",")[0]
    symbols_lst = text_to_symbols(texts, resource_dir, speaker, lang)
    symbols_file = os.path.join(output_dir, "symbols.lst")
    with open(symbols_file, "w") as symbol_data:
        for symbol in symbols_lst:
            symbol_data.write(symbol)

    logging.info("AM is infering...")
    am_infer(symbols_file, am_ckpt, output_dir)

    logging.info("Vocoder is infering...")
    hifigan_infer(os.path.join(output_dir, "feat"), voc_ckpt, output_dir)

    concat_process(output_dir, os.path.join(output_dir, "res_wavs"))

    logging.info("Text to wav finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text to wav")
    parser.add_argument("--txt", type=str, required=True, help="Path to text file")
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to output directory"
    )
    parser.add_argument(
        "--res_zip", type=str, required=True, help="Path to resource zip file"
    )
    parser.add_argument(
        "--am_ckpt", type=str, required=True, help="Path to am ckpt file"
    )
    parser.add_argument(
        "--voc_ckpt", type=str, required=True, help="Path to voc ckpt file"
    )
    parser.add_argument(
        "--speaker",
        type=str,
        required=False,
        default=None,
        help="The speaker name, default is the first speaker",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="PinYin",
        help="""The language of the text, default is PinYin, other options are:
        English,
        British,
        ZhHK,
        WuuShanghai,
        Sichuan,
        Indonesian,
        Malay,
        Filipino,
        Vietnamese,
        Korean,
        Russian
        """,
    )
    args = parser.parse_args()
    text_to_wav(
        args.txt,
        args.output_dir,
        args.res_zip,
        args.am_ckpt,
        args.voc_ckpt,
        args.speaker,
        args.lang,
    )
