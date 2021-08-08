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
        self.music_max = np.load(os.path.join(audio_dir, 'max.npy'))
        self.music_min = np.load(os.path.join(audio_dir, 'min.npy'))

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
        min_max = (mfcc.T - self.music_min) / (self.music_max - self.music_min)
        norm_mfcc = 2 * min_max - 1
        assert norm_mfcc.min() >= -1 and norm_mfcc.max() <= 1
        
        return 2 * min_max - 1

    def revert(self, mfcc):
        min_max = (mfcc + 1) / 2
        mfcc = min_max * (self.music_max - self.music_min) + self.music_min
        print(mfcc.shape)
        
        return mfcc

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