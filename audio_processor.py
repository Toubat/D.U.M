import os 
import numpy as np
import librosa
import librosa.display
from scipy.io.wavfile import write


class AudioToMFCCs():
    def __init__(self, sample_rate, n_mfcc):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc

    def audio_to_mfcc(self, audio_dir):
        audio, sample_rate = librosa.load(audio_dir, sr=self.sample_rate)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=self.n_mfcc)

        return mfccs

    def mfcc_to_audio(self, mfccs, audio_dir):
        audio = librosa.feature.inverse.mfcc_to_audio(mfccs, sr=self.sample_rate)
        write(audio_dir, self.sample_rate, audio.squeeze())

        return audio
        


def main():
    # The following line of code is only needed for windows 
    converter = AudioToMFCCs(sample_rate=48000, n_mfcc=128)    
    directory = './refined_wav'
    mfcc_directory = './audio_mfcc'

    for filename in os.listdir(directory):
        mfccs = converter.audio_to_mfcc('/'.join([directory, filename]))
        np.save('/'.join([mfcc_directory,'.'.join([os.path.splitext(filename)[0],'npy'])]), mfccs, allow_pickle=True, fix_imports=True)
        # audio = converter.mfcc_to_audio(mfccs, 'test.wav')
        assert mfccs.shape == (128, 938)

    return 0
