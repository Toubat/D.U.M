import torch
import os
import numpy as np
from torch.utils.data import Dataset


class GestureMusicDataset(Dataset):
    def __init__(self, gesture_dir, audio_dir, padding_len):
        self.gesture_dir = gesture_dir
        self.audio_dir = audio_dir
        self.gestures = os.listdir(gesture_dir)
        self.audios = audio_dir
        self.padding_len = padding_len

    def __len__(self):
        return len(self.gestures)

    def __getitem__(self, idx):
        gesture_path = os.path.join(self.gesture_dir, self.gestures[idx])
        music_path = os.path.join(self.audio_dir, self.get_music_name(idx))
        gesture_keypoints = np.load(gesture_path)
        gesture_keypoints_pad = self.get_gesture_padding(gesture_keypoints)
        music_spectrogram = np.load(music_path)

        return gesture_keypoints_pad, music_spectrogram

    def get_music_name(self, idx):
        start = self.gestures[idx].find('_m') + 1
        return f'{self.gestures[idx][start:start+4]}.npy'

    def get_gesture_padding(self, gesture_keypoints):
        gesture_len = gesture_keypoints.shape[0]
        if self.padding_len - gesture_len < 0:
            return gesture_keypoints[:self.padding_len, :]
        elif self.padding_len - gesture_len > 0:
            zeros = np.zeros((self.padding_len - gesture_len, gesture_keypoints.shape[1]))
            return np.concatenate((gesture_keypoints, zeros), axis=0)
        else:
            return gesture_keypoints


if __name__ == '__main__':
    lens = []
    dataset = GestureMusicDataset(
        gesture_dir='./video', 
        audio_dir='./npy_audio', 
        padding_len=512
    )
    for i, (gesture, music) in enumerate(dataset):
        print(music)
        break