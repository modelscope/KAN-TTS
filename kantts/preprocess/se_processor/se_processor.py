import torch
import torchaudio
import numpy as np
import os
import torchaudio.compliance.kaldi as Kaldi
from .D_TDNN import DTDNN
import logging
import argparse
from glob import glob


logging.basicConfig(
    format="%(asctime)s %(levelname)-4s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.DEBUG,
)

class SpeakerEmbeddingProcessor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.min_wav_length = self.sample_rate * 30 * 10 / 1000

        self.pcm_dict = {}
        self.mfcc_dict = {}
        self.se_list = []

    def process(self, src_voice_dir, se_model):
        logging.info("[SpeakerEmbeddingProcessor] Speaker embedding extractor started")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DTDNN()
        try:
            if os.path.basename(se_model) == "se.model":
                model.load_state_dict(torch.load(se_model, map_location=device))
            else:
                raise Exception("[SpeakerEmbeddingProcessor] se model loading error!!!")
        except Exception as e:
            logging.info(e)
            if os.path.basename(se_model) == 'se.onnx':
                logging.info("[SpeakerEmbeddingProcessor] please update your se model to ensure that the version is greater than or equal to 1.0.5")
            sys.exit()
        model.eval()
        model.to(device)

        wav_dir = os.path.join(src_voice_dir, "wav")
        se_dir = os.path.join(src_voice_dir, "se")
        se_average_file = os.path.join(se_dir, "se.npy")

        os.makedirs(se_dir, exist_ok=True)

        wav_files = glob(os.path.join(wav_dir, '*.wav'))


        for wav_file in wav_files:
            basename = os.path.splitext(os.path.basename(wav_file))[0]
            se_file = os.path.join(se_dir, basename + '.npy')
             
            wav, fs = torchaudio.load(wav_file)
            assert wav.shape[0] == 1
            assert fs == 16000

            if wav.shape[1] < self.min_wav_length:
                continue

            fbank_feat = Kaldi.fbank(wav, num_mel_bins=80)
            
            feat = fbank_feat - fbank_feat.mean(dim=0, keepdim=True)
            feat = feat.unsqueeze(0).to(device)
            
            speaker_embedding = model(feat)
            speaker_embedding = speaker_embedding.squeeze().cpu().detach().numpy()
            speaker_embedding = np.expand_dims(speaker_embedding,  axis=0)
            
            
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
    parser.add_argument('--se_model', required=True)
    args = parser.parse_args()

    sep = SpeakerEmbeddingProcessor()
    sep.process(args.src_voice_dir, args.se_onnx)