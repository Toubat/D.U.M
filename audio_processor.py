import librosa
import librosa.display
import matplotlib.pyplot as plt

import os 

import torch
import torchaudio
import torchaudio.transforms as T

import numpy as np
from scipy.io.wavfile import write


class AudioToMelSpectrogram():
    def __init__(self, sample_rate, n_fft, hop_length, n_mels):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def audio_to_mel(self, audio_file='./refined_wav/mBR0.wav'):
        waveform, sample_rate = torchaudio.load(audio_file)

        resampler = T.Resample(sample_rate, self.sample_rate)
        mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )

        waveform = resampler(waveform)
        # (2, num_samples) -> (1, num_samples)
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        mel_sgram = mel_spectrogram(waveform)
        mel_sgram = librosa.amplitude_to_db(mel_sgram[0].numpy())

        # librosa.display.specshow(mel_sgram, sr=22050, x_axis='time', y_axis='mel')  
        # plt.colorbar(format='%+2.0f dB')

        return mel_sgram

    def mel_to_audio(self, mel_sgram):
        # still under development (reducing noise needed!)
        specgram = librosa.db_to_amplitude(mel_sgram)
        audio = librosa.feature.inverse.mel_to_audio(specgram, n_fft=self.n_fft, hop_length=self.hop_length).reshape(1, -1)

        return audio
        


def main():
    # The following line of code is only needed for windows 
    #torchaudio.set_audio_backend("soundfile")
    converter = AudioToMelSpectrogram(sample_rate=22050, n_fft=1024, hop_length=512, n_mels=128)    


    directory = './refined_wav'
    mel_directory = './npy_audio'
    for filename in os.listdir(directory):
        mel_sgram = converter.audio_to_mel('/'.join([directory,filename]))
        np.save('/'.join([mel_directory,'.'.join([os.path.splitext(filename)[0],'npy'])]), mel_sgram, allow_pickle=True, fix_imports=True)
    
    audio = converter.mel_to_audio(mel_sgram)
    #write .wav to a specific file (not needed for now)
    #write('test.wav', 22050, audio.squeeze())
    return 0
