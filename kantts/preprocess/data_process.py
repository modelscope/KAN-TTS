import logging
import os
import sys
import argparse
import yaml
import time

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # NOQA: E402
sys.path.insert(0, os.path.dirname(ROOT_PATH))  # NOQA: E402

try:
    from kantts.preprocess.audio_processor.audio_processor import AudioProcessor
    from kantts.preprocess.script_convertor.TextScriptConvertor import (
        TextScriptConvertor,
    )
    from kantts.datasets.dataset import AM_Dataset, Voc_Dataset
    from kantts.utils.log import logging_to_file, get_git_revision_hash
except ImportError:
    raise ImportError("Please install kantts.")

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.DEBUG,
)

LANGUAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "languages")

languages = {
    "PinYin": {
        "phoneset_path": "PhoneSet.xml",
        "posset_path": "PosSet.xml",
        "f2t_map_path": "En2ChPhoneMap.txt",
        "s2p_map_path": "py2phoneMap.txt",
    }
}


def gen_metafile(
    voice_output_dir,
    split_ratio=0.98,
):

    voc_train_meta = os.path.join(voice_output_dir, "train.lst")
    voc_valid_meta = os.path.join(voice_output_dir, "valid.lst")
    if not os.path.exists(voc_train_meta) or not os.path.exists(voc_valid_meta):
        Voc_Dataset.gen_metafile(
            os.path.join(voice_output_dir, "wav"), voice_output_dir, split_ratio
        )
        logging.info("Voc metafile generated.")

    raw_metafile = os.path.join(voice_output_dir, "raw_metafile.txt")
    am_train_meta = os.path.join(voice_output_dir, "am_train.lst")
    am_valid_meta = os.path.join(voice_output_dir, "am_valid.lst")
    if not os.path.exists(am_train_meta) or not os.path.exists(am_valid_meta):
        AM_Dataset.gen_metafile(raw_metafile, voice_output_dir, split_ratio)
        logging.info("AM metafile generated.")


#  TODO: Zh-CN as default
def process_mit_style_data(
    voice_input_dir,
    voice_output_dir,
    audio_config,
    speaker_name=None,
    skip_script=False,
):
    targetLang = "PinYin"
    foreignLang = "EnUS"
    #  TODO: check if the vocie is supported
    emo_tag_path = None

    os.makedirs(voice_output_dir, exist_ok=True)
    logging_to_file(os.path.join(voice_output_dir, "data_process_stdout.log"))

    phoneset_path = os.path.join(
        LANGUAGES_DIR, targetLang, languages[targetLang]["phoneset_path"]
    )
    posset_path = os.path.join(
        LANGUAGES_DIR, targetLang, languages[targetLang]["posset_path"]
    )
    f2t_map_path = os.path.join(
        LANGUAGES_DIR, targetLang, languages[targetLang]["f2t_map_path"]
    )
    s2p_map_path = os.path.join(
        LANGUAGES_DIR, targetLang, languages[targetLang]["s2p_map_path"]
    )

    if speaker_name is None:
        speaker_name = os.path.basename(voice_input_dir)

    if audio_config is not None:
        with open(audio_config, "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)

    config["create_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    config["git_revision_hash"] = get_git_revision_hash()

    with open(os.path.join(voice_output_dir, "audio_config.yaml"), "w") as f:
        yaml.dump(config, f, Dumper=yaml.Dumper, default_flow_style=None)

    if skip_script:
        logging.info("Skip script conversion")
    raw_metafile = None
    #  Script processor
    if not skip_script:
        tsc = TextScriptConvertor(
            phoneset_path,
            posset_path,
            targetLang,
            foreignLang,
            f2t_map_path,
            s2p_map_path,
            emo_tag_path,
            speaker_name,
        )
        tsc.process(
            os.path.join(voice_input_dir, "prosody", "prosody.txt"),
            os.path.join(voice_output_dir, "Script.xml"),
            os.path.join(voice_output_dir, "raw_metafile.txt"),
        )
        raw_metafile = os.path.join(voice_output_dir, "raw_metafile.txt")

    #  Audio processor
    ap = AudioProcessor(config["audio_config"])
    ap.process(
        voice_input_dir,
        voice_output_dir,
        raw_metafile,
    )

    logging.info("Processing done.")

    # Generate Voc&AM metafile
    # TODO: train/valid ratio setting
    gen_metafile(voice_output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset preprocessor")
    parser.add_argument("--voice_input_dir", type=str, required=True)
    parser.add_argument("--voice_output_dir", type=str, required=True)
    parser.add_argument("--audio_config", type=str, required=True)
    parser.add_argument("--speaker", type=str, default=None, help="speaker")
    parser.add_argument(
        "--skip_script", action="store_true", help="skip script converting"
    )
    args = parser.parse_args()

    process_mit_style_data(
        args.voice_input_dir,
        args.voice_output_dir,
        args.audio_config,
        args.speaker,
        args.skip_script,
    )
