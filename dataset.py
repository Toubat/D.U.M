import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader


class GestureMusicDataset(Dataset):
    def __init__(self, gesture_dir, audio_dir, padding_len):
        self.gesture_dir = gesture_dir
        self.audio_dir = audio_dir
        self.gestures = os.listdir(gesture_dir)
        self.audios = audio_dir
        self.padding_len = padding_len
        self.music_mean = np.load(os.path.join(audio_dir, 'mean.npy'))
        self.music_std = np.load(os.path.join(audio_dir, 'std.npy'))

    def __len__(self):
        return len(self.gestures)

    def __getitem__(self, idx):
        gesture_path = os.path.join(self.gesture_dir, self.gestures[idx])
        music_path = os.path.join(self.audio_dir, self.get_music_name(idx))
        gesture_keypoints = np.load(gesture_path)
        gesture_keypoints_pad = self.get_gesture_padding(gesture_keypoints)
        music_mfcc = self.normalize(np.load(music_path))

        return torch.tensor(gesture_keypoints_pad), torch.tensor(music_mfcc)

    def normalize(self, mfcc):
        norm_mfcc = (mfcc.T - self.music_mean) / self.music_std
        return norm_mfcc

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
    dataset = GestureMusicDataset(gesture_dir='./video', audio_dir='./audio_mfcc', padding_len=512)
    # Create a DataLoader to process dataset in batches
    data_loader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=2)
    for i, (gestures, musics) in enumerate(data_loader):
        if i == 5:
            break
        print(gestures.size(), musics.size())