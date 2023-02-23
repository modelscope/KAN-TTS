import os
import numpy as np
from numpy import dot
from numpy.linalg import norm
import librosa
import argparse
import onnxruntime
import torch
from glob import glob
from .core.feature import (
    compute_mfcc_feats,
    apply_cmvn_sliding,
)
from .core.ivector import compute_vad
import logging


logging.basicConfig(
    format="%(asctime)s %(levelname)-4s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.DEBUG,
)

def extract_features(wav):
    data = (wav * 32768).astype(np.int16)    
    raw_mfcc = compute_mfcc_feats(data, dither=0.0, energy_floor=0.1, sample_frequency=16000, frame_length=25, frame_shift=10, low_freq=20, high_freq=-400, num_mel_bins=30, num_ceps=30, snip_edges=False)
    log_energy = raw_mfcc[:, 0]
    vad = compute_vad(log_energy, energy_threshold=5.5, energy_mean_scale=0.5, frames_context=2, proportion_threshold=0.12)
    mfcc = apply_cmvn_sliding(raw_mfcc, window=300, center=True)#[vad]
    return(np.float32(mfcc))

def extract_se(sess, inputs_mfcc):
    feat = inputs_mfcc.T 
    feat = torch.from_numpy(feat)
    feat = torch.unsqueeze(feat, 0)  
    outputs = sess.run(
        ['output'], 
        {'input': feat.data.cpu().numpy()}
    )
    speaker_embedding = outputs[0]
    return speaker_embedding

def read_scp(wav2sv):
    wavlst = []
    svlst = []
    f = open(wav2sv)
    for line in f.readlines():
        line = line.strip().split(' ')
        wavlst.append(line[0])
        svlst.append(line[1])
    return(wavlst, svlst)

class SpeakerEmbeddingProcessor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.min_wav_length = self.sample_rate * 30 * 10 / 1000

        self.pcm_dict = {}
        self.mfcc_dict = {}
        self.se_list = []

    def process(self, src_voice_dir, se_onnx):
        logging.info("[SpeakerEmbeddingProcessor] Speaker embedding extractor started")

        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = 1
        sess = onnxruntime.InferenceSession(se_onnx, sess_options=opts)

        wav_dir = os.path.join(src_voice_dir, "wav")
        se_dir = os.path.join(src_voice_dir, "se")
        se_average_file = os.path.join(se_dir, "se.npy")

        os.makedirs(se_dir, exist_ok=True)

        wav_files = glob(os.path.join(wav_dir, '*.wav'))

        for wav_file in wav_files:
            basename = os.path.splitext(os.path.basename(wav_file))[0]
            se_file = os.path.join(se_dir, basename + '.npy')
             
            wav, _ = librosa.load(wav_file, sr=self.sample_rate)
            if len(wav) < self.min_wav_length:
                continue

            feat = extract_features(wav)
            feat = feat.T 
            feat = torch.from_numpy(feat)
            feat = torch.unsqueeze(feat, 0)  
            outputs = sess.run(
                ['output'], 
                {'input': feat.data.cpu().numpy()}
            )
            speaker_embedding = outputs[0]
            np.save(se_file, speaker_embedding)
            self.se_list.append(speaker_embedding)
        self.se_average = np.expand_dims(
            np.mean(
                np.concatenate(self.se_list, axis=0), 
                axis=0
            ), 
            axis=0
        )
        np.save(se_average_file, self.se_average)

        logging.info("[SpeakerEmbeddingProcessor] Speaker embedding extracted successfully!")
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Speaker Embedding Processor")
    parser.add_argument("--src_voice_dir", type=str, required=True)    
    parser.add_argument('--se_onnx', required=True)
    args = parser.parse_args()

    sep = SpeakerEmbeddingProcessor()
    sep.process(args.src_voice_dir, args.se_onnx)