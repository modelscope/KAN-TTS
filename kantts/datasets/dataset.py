import os
import torch
import glob
import logging
from multiprocessing import Manager
import librosa
import numpy as np
import random
from tqdm import tqdm
from kantts.utils.ling_unit.ling_unit import KanTtsLinguisticUnit


class Padder(object):
    def __init__(self):
        super(Padder, self).__init__()
        pass

    def _pad1D(self, x, length, pad):
        return np.pad(x, (0, length - x.shape[0]), mode="constant", constant_values=pad)

    def _pad2D(self, x, length, pad):
        return np.pad(
            x, [(0, length - x.shape[0]), (0, 0)], mode="constant", constant_values=pad
        )

    def _pad_durations(self, duration, max_in_len, max_out_len):
        framenum = np.sum(duration)
        symbolnum = duration.shape[0]
        if framenum < max_out_len:
            padframenum = max_out_len - framenum
            duration = np.insert(duration, symbolnum, values=padframenum, axis=0)
            duration = np.insert(
                duration,
                symbolnum + 1,
                values=[0] * (max_in_len - symbolnum - 1),
                axis=0,
            )
        else:
            if symbolnum < max_in_len:
                duration = np.insert(
                    duration, symbolnum, values=[0] * (max_in_len - symbolnum), axis=0
                )
        return duration

    def _round_up(self, x, multiple):
        remainder = x % multiple
        return x if remainder == 0 else x + multiple - remainder

    def _prepare_scalar_inputs(self, inputs, max_len, pad):
        return torch.from_numpy(
            np.stack([self._pad1D(x, max_len, pad) for x in inputs])
        )

    def _prepare_targets(self, targets, max_len, pad):
        return torch.from_numpy(
            np.stack([self._pad2D(t, max_len, pad) for t in targets])
        ).float()

    def _prepare_durations(self, durations, max_in_len, max_out_len):
        return torch.from_numpy(
            np.stack(
                [self._pad_durations(t, max_in_len, max_out_len) for t in durations]
            )
        ).long()


class Voc_Dataset(torch.utils.data.Dataset):
    """
    provide (mel, audio) data pair
    """

    def __init__(
        self,
        metafile,
        root_dir,
        sampling_rate=24000,
        n_fft=1024,
        hop_length=240,
        allow_cache=False,
        batch_max_steps=20480,
    ):
        self.meta = []
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.batch_max_steps = batch_max_steps
        self.batch_max_frames = self.batch_max_steps // self.hop_length
        self.aux_context_window = 0  # TODO: make it configurable
        self.start_offset = self.aux_context_window
        self.end_offset = -(self.batch_max_frames + self.aux_context_window)

        if not isinstance(metafile, list):
            metafile = [metafile]
        if not isinstance(root_dir, list):
            root_dir = [root_dir]

        for meta_file, data_dir in zip(metafile, root_dir):
            if not os.path.exists(meta_file):
                logging.error("meta file not found: {}".format(meta_file))
                raise ValueError(
                    "[Voc_Dataset] meta file: {} not found".format(meta_file)
                )
            if not os.path.exists(data_dir):
                logging.error("data directory not found: {}".format(data_dir))
                raise ValueError(
                    "[Voc_Dataset] data dir: {} not found".format(data_dir)
                )
            self.meta.extend(self.load_meta(meta_file, data_dir))

        #  Load from training data directory
        if len(self.meta) == 0 and isinstance(root_dir, str):
            wav_dir = os.path.join(root_dir, "wav")
            mel_dir = os.path.join(root_dir, "mel")
            if not os.path.exists(wav_dir) or not os.path.exists(mel_dir):
                raise ValueError("wav or mel directory not found")
            self.meta.extend(self.load_meta_from_dir(wav_dir, mel_dir))
        elif len(self.meta) == 0 and isinstance(root_dir, list):
            for d in root_dir:
                wav_dir = os.path.join(d, "wav")
                mel_dir = os.path.join(d, "mel")
                if not os.path.exists(wav_dir) or not os.path.exists(mel_dir):
                    raise ValueError("wav or mel directory not found")
                self.meta.extend(self.load_meta_from_dir(wav_dir, mel_dir))

        self.allow_cache = allow_cache
        if allow_cache:
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(self.meta))]

    @staticmethod
    def gen_metafile(wav_dir, out_dir, split_ratio=0.98):
        wav_files = glob.glob(os.path.join(wav_dir, "*.wav"))
        random.shuffle(wav_files)
        num_train = int(len(wav_files) * split_ratio) - 1
        with open(os.path.join(out_dir, "train.lst"), "w") as f:
            for wav_file in wav_files[:num_train]:
                f.write("{}\n".format(os.path.splitext(os.path.basename(wav_file))[0]))

        with open(os.path.join(out_dir, "valid.lst"), "w") as f:
            for wav_file in wav_files[num_train:]:
                f.write("{}\n".format(os.path.splitext(os.path.basename(wav_file))[0]))

    def load_meta(self, metafile, data_dir):
        with open(metafile, "r") as f:
            lines = f.readlines()
        wav_dir = os.path.join(data_dir, "wav")
        mel_dir = os.path.join(data_dir, "mel")
        if not os.path.exists(wav_dir) or not os.path.exists(mel_dir):
            raise ValueError("wav or mel directory not found")
        items = []
        logging.info("Loading metafile...")
        for name in tqdm(lines):
            name = name.strip()
            mel_file = os.path.join(mel_dir, name + ".npy")
            wav_file = os.path.join(wav_dir, name + ".wav")
            if os.path.exists(mel_file) and os.path.exists(wav_file):
                items.append((wav_file, mel_file))
        return items

    def load_meta_from_dir(self, wav_dir, mel_dir):
        wav_files = glob.glob(os.path.join(wav_dir, "*.wav"))
        items = []
        for wav_file in wav_files:
            mel_file = os.path.join(mel_dir, os.path.basename(wav_file))
            if os.path.exists(mel_file):
                items.append((wav_file, mel_file))
        return items

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]

        wav_file, mel_file = self.meta[idx]

        wav_data = librosa.core.load(wav_file, sr=self.sampling_rate)[0]
        mel_data = np.load(mel_file)

        # make sure the audio length and feature length are matched
        wav_data = np.pad(wav_data, (0, self.n_fft), mode="reflect")
        wav_data = wav_data[: len(mel_data) * self.hop_length]
        assert len(mel_data) * self.hop_length == len(wav_data)

        if self.allow_cache:
            self.caches[idx] = (wav_data, mel_data)
        return (wav_data, mel_data)

    def collate_fn(self, batch):
        wav_data, mel_data = [item[0] for item in batch], [item[1] for item in batch]
        mel_lengths = [len(mel) for mel in mel_data]

        start_frames = np.array(
            [
                np.random.randint(self.start_offset, length + self.end_offset)
                for length in mel_lengths
            ]
        )

        wav_start = start_frames * self.hop_length
        wav_end = wav_start + self.batch_max_steps

        # aux window works as padding
        mel_start = start_frames - self.aux_context_window
        mel_end = mel_start + self.batch_max_frames + self.aux_context_window

        wav_batch = [
            x[start:end] for x, start, end in zip(wav_data, wav_start, wav_end)
        ]
        mel_batch = [
            c[start:end] for c, start, end in zip(mel_data, mel_start, mel_end)
        ]

        # (B, 1, T)
        wav_batch = torch.tensor(np.asarray(wav_batch), dtype=torch.float32).unsqueeze(
            1
        )
        # (B, C, T)
        mel_batch = torch.tensor(np.asarray(mel_batch), dtype=torch.float32).transpose(
            2, 1
        )
        return wav_batch, mel_batch


def get_voc_datasets(
    root_dir,
    sampling_rate,
    n_fft,
    hop_length,
    allow_cache,
    batch_max_steps,
    split_ratio=0.98,
):
    if isinstance(root_dir, str):
        root_dir = [root_dir]
    train_meta_lst = []
    valid_meta_lst = []
    for data_dir in root_dir:
        train_meta = os.path.join(data_dir, "train.lst")
        valid_meta = os.path.join(data_dir, "valid.lst")
        if not os.path.exists(train_meta) or not os.path.exists(valid_meta):
            Voc_Dataset.gen_metafile(
                os.path.join(data_dir, "wav"), data_dir, split_ratio
            )
        train_meta_lst.append(train_meta)
        valid_meta_lst.append(valid_meta)
    train_dataset = Voc_Dataset(
        train_meta_lst,
        root_dir,
        sampling_rate,
        n_fft,
        hop_length,
        allow_cache,
        batch_max_steps,
    )

    valid_dataset = Voc_Dataset(
        valid_meta_lst,
        root_dir,
        sampling_rate,
        n_fft,
        hop_length,
        allow_cache,
        batch_max_steps,
    )

    return train_dataset, valid_dataset


class AM_Dataset(torch.utils.data.Dataset):
    """
    provide (ling, emo, speaker, mel) pair
    """

    def __init__(
        self,
        config,
        metafile,
        root_dir,
        allow_cache=False,
    ):
        self.meta = []
        self.config = config

        if not isinstance(metafile, list):
            metafile = [metafile]
        if not isinstance(root_dir, list):
            root_dir = [root_dir]

        for meta_file, data_dir in zip(metafile, root_dir):
            if not os.path.exists(meta_file):
                logging.error("meta file not found: {}".format(meta_file))
                raise ValueError(
                    "[AM_Dataset] meta file: {} not found".format(meta_file)
                )
            if not os.path.exists(data_dir):
                logging.error("data dir not found: {}".format(data_dir))
                raise ValueError("[AM_Dataset] data dir: {} not found".format(data_dir))
            self.meta.extend(self.load_meta(meta_file, data_dir))

        self.allow_cache = allow_cache

        self.ling_unit = KanTtsLinguisticUnit(config)
        self.padder = Padder()

        self.r = self.config["Model"]["KanTtsSAMBERT"]["params"]["outputs_per_step"]
        #  TODO: feat window

        if allow_cache:
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(self.meta))]

    def __len__(self):
        return len(self.meta)

    #  TODO: implement __getitem__
    def __getitem__(self, idx):
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]

        ling_txt, mel_file, dur_file, f0_file, energy_file = self.meta[idx]

        ling_data = self.ling_unit.encode_symbol_sequence(ling_txt)
        mel_data = np.load(mel_file)
        dur_data = np.load(dur_file)
        f0_data = np.load(f0_file)
        energy_data = np.load(energy_file)

        #  make sure the audio length and feature length are matched
        #  assert len(mel_data) == np.sum(
        #          dur_data
        #  ), "audio and feature length not matched: {} vs {}".format(mel_file, dur_file)
        #  assert len(ling_data[0]) == len(
        #      dur_data
        #  ), "linguistic and feature length not matched: {}, {} v.s {}".format(
        #          dur_file, len(ling_data[0]), len(dur_data))
        #  assert len(f0_data) == len(
        #      dur_data
        #  ), "f0 and dur length not matched: {} vs {}".format(f0_file, dur_file)
        #  assert len(energy_data) == len(
        #      dur_data
        #  ), "energy and dur length not matched: {} vs {}".format(energy_file, dur_file)

        if self.allow_cache:
            self.caches[idx] = (ling_data, mel_data, dur_data, f0_data, energy_data)

        return (ling_data, mel_data, dur_data, f0_data, energy_data)

    def load_meta(self, metafile, data_dir):
        with open(metafile, "r") as f:
            lines = f.readlines()

        if self.config["audio_config"]["trim_silence"]:
            mel_dir = os.path.join(data_dir, "trim_mel")
        else:
            mel_dir = os.path.join(data_dir, "mel")
        dur_dir = os.path.join(data_dir, "duration")
        f0_dir = os.path.join(data_dir, "f0")
        energy_dir = os.path.join(data_dir, "energy")

        items = []
        logging.info("Loading metafile...")
        for line in tqdm(lines):
            line = line.strip()
            index, ling_txt = line.split("\t")
            mel_file = os.path.join(mel_dir, index + ".npy")
            dur_file = os.path.join(dur_dir, index + ".npy")
            f0_file = os.path.join(f0_dir, index + ".npy")
            energy_file = os.path.join(energy_dir, index + ".npy")
            if (
                os.path.exists(mel_file)
                and os.path.exists(dur_file)
                and os.path.exists(f0_file)
                and os.path.exists(energy_file)
            ):
                items.append(
                    (
                        ling_txt,
                        mel_file,
                        dur_file,
                        f0_file,
                        energy_file,
                    )
                )

        return items

    @staticmethod
    def gen_metafile(raw_meta_file, out_dir, split_ratio=0.98):
        with open(raw_meta_file, "r") as f:
            lines = f.readlines()
        random.shuffle(lines)
        num_train = int(len(lines) * split_ratio) - 1
        with open(os.path.join(out_dir, "am_train.lst"), "w") as f:
            for line in lines[:num_train]:
                f.write(line)

        with open(os.path.join(out_dir, "am_valid.lst"), "w") as f:
            for line in lines[num_train:]:
                f.write(line)

    #  TODO: implement collate_fn
    def collate_fn(self, batch):
        data_dict = {}

        max_input_length = max((len(x[0][0]) for x in batch))

        # pure linguistic info: sy|tone|syllable_flag|word_segment
        lfeat_type = self.ling_unit._lfeat_type_list[0]
        inputs_sy = self.padder._prepare_scalar_inputs(
            [x[0][0] for x in batch],
            max_input_length,
            self.ling_unit._sub_unit_pad[lfeat_type],
        ).long()
        # tone
        lfeat_type = self.ling_unit._lfeat_type_list[1]
        inputs_tone = self.padder._prepare_scalar_inputs(
            [x[0][1] for x in batch],
            max_input_length,
            self.ling_unit._sub_unit_pad[lfeat_type],
        ).long()

        # syllable_flag
        lfeat_type = self.ling_unit._lfeat_type_list[2]
        inputs_syllable_flag = self.padder._prepare_scalar_inputs(
            [x[0][2] for x in batch],
            max_input_length,
            self.ling_unit._sub_unit_pad[lfeat_type],
        ).long()

        # word_segment
        lfeat_type = self.ling_unit._lfeat_type_list[3]
        inputs_ws = self.padder._prepare_scalar_inputs(
            [x[0][3] for x in batch],
            max_input_length,
            self.ling_unit._sub_unit_pad[lfeat_type],
        ).long()

        # emotion category
        lfeat_type = self.ling_unit._lfeat_type_list[4]
        data_dict["input_emotions"] = self.padder._prepare_scalar_inputs(
            [x[0][4] for x in batch],
            max_input_length,
            self.ling_unit._sub_unit_pad[lfeat_type],
        ).long()

        # speaker category
        lfeat_type = self.ling_unit._lfeat_type_list[5]
        data_dict["input_speakers"] = self.padder._prepare_scalar_inputs(
            [x[0][5] for x in batch],
            max_input_length,
            self.ling_unit._sub_unit_pad[lfeat_type],
        ).long()

        data_dict["input_lings"] = torch.stack(
            [inputs_sy, inputs_tone, inputs_syllable_flag, inputs_ws], dim=2
        )
        data_dict["valid_input_lengths"] = torch.as_tensor(
            [len(x[0][0]) - 1 for x in batch], dtype=torch.long
        )  # 输入的symbol sequence会在后面拼一个“~”，影响duration计算，所以把length-1
        data_dict["valid_output_lengths"] = torch.as_tensor(
            [len(x[1]) for x in batch], dtype=torch.long
        )

        max_output_length = torch.max(data_dict["valid_output_lengths"]).item()
        max_output_round_length = self.padder._round_up(max_output_length, self.r)

        data_dict["mel_targets"] = self.padder._prepare_targets(
            [x[1] for x in batch], max_output_round_length, 0.0
        )
        data_dict["durations"] = self.padder._prepare_durations(
            [x[2] for x in batch], max_input_length, max_output_round_length
        )

        data_dict["pitch_contours"] = self.padder._prepare_scalar_inputs(
            [x[3] for x in batch], max_input_length, 0.0
        ).float()
        data_dict["energy_contours"] = self.padder._prepare_scalar_inputs(
            [x[4] for x in batch], max_input_length, 0.0
        ).float()

        return data_dict


#  TODO: implement get_am_datasets
def get_am_datasets(
    metafile,
    root_dir,
    config,
    allow_cache,
    split_ratio=0.98,
):
    if not isinstance(root_dir, list):
        root_dir = [root_dir]
    if not isinstance(metafile, list):
        metafile = [metafile]

    train_meta_lst = []
    valid_meta_lst = []

    for raw_metafile, data_dir in zip(metafile, root_dir):
        train_meta = os.path.join(data_dir, "am_train.lst")
        valid_meta = os.path.join(data_dir, "am_valid.lst")
        if not os.path.exists(train_meta) or not os.path.exists(valid_meta):
            AM_Dataset.gen_metafile(raw_metafile, data_dir, split_ratio)
        train_meta_lst.append(train_meta)
        valid_meta_lst.append(valid_meta)

    train_dataset = AM_Dataset(config, train_meta_lst, root_dir, allow_cache)

    valid_dataset = AM_Dataset(config, valid_meta_lst, root_dir, allow_cache)

    return train_dataset, valid_dataset
