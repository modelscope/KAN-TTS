import logging
import os
import sys
import argparse
import yaml
import time
import zipfile

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # NOQA: E402
sys.path.insert(0, os.path.dirname(ROOT_PATH))  # NOQA: E402

try:
    from kantts.datasets.dataset import BERT_Text_Dataset
    from kantts.utils.log import logging_to_file, get_git_revision_hash
    from kantts.utils.ling_unit import text_to_mit_symbols as text_to_symbols
except ImportError:
    raise ImportError("Please install kantts.")

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)


def gen_metafile(
    output_dir,
    split_ratio=0.98,
):
    raw_metafile = os.path.join(output_dir, "raw_metafile.txt")
    bert_train_meta = os.path.join(output_dir, "bert_train.lst")
    bert_valid_meta = os.path.join(output_dir, "bert_valid.lst")
    if not os.path.exists(
            bert_train_meta) or not os.path.exists(bert_valid_meta):
        BERT_Text_Dataset.gen_metafile(raw_metafile, output_dir, split_ratio)
        logging.info("BERT Text metafile generated.")

#  TODO: Zh-CN as default
def process_mit_style_data(
    text_file,
    resources_zip_file,
    output_dir,
):
    os.makedirs(output_dir, exist_ok=True)
    logging_to_file(os.path.join(output_dir, "data_process_stdout.log"))

    resource_root_dir = os.path.dirname(resources_zip_file)
    resource_dir = os.path.join(resource_root_dir, "resource")

    if not os.path.exists(resource_dir):
        logging.info("Extracting resources...")
        with zipfile.ZipFile(resources_zip_file, "r") as zip_ref:
            zip_ref.extractall(resource_root_dir)

    with open(text_file, "r") as text_data:
        texts = text_data.readlines()

    logging.info("Converting text to symbols...")
    symbols_lst = text_to_symbols(texts, resource_dir, "F7")
    symbols_file = os.path.join(output_dir, "raw_metafile.txt")
    with open(symbols_file, "w") as symbol_data:
        for symbol in symbols_lst:
            symbol_data.write(symbol)

    logging.info("Processing done.")

    # Generate BERT Text metafile
    # TODO: train/valid ratio setting
    gen_metafile(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset preprocessor")
    parser.add_argument("--text_file", type=str, required=True)
    parser.add_argument("--resources_zip_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    process_mit_style_data(
        args.text_file,
        args.resources_zip_file,
        args.output_dir,
    )

