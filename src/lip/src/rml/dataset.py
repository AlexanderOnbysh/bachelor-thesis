import os
from glob import glob

import cv2
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from constants import CHAR_SET
import torch
import re 


def load_video(video_path):
    """reshape the (# of frame, 120, 120) video to (batch size, # from frame, 120, 120)"""
    cap = cv2.VideoCapture(video_path)
    buffer = []

    has_content, frame = cap.read()
    count = 0
    while has_content:
        # Fetch every 5 frames.
        count += 1
        if count % 5 == 0:
            has_content, frame = cap.read()
            continue

        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (120, 120)).reshape(1, 120, 120)
        buffer.append(gray)
        has_content, frame = cap.read()
    try:
        # make sure the total frame can be dived by 5 by dropping the extra frames.
        buffer = buffer[:int(len(buffer)/5)*5]
        image_tensor = torch.from_numpy(np.concatenate(buffer, axis=0))
        # torch believes image_tensor is a BytesTensor. let's convert it to int.
        results = torch.zeros(len(buffer), 120, 120)
        results[:, :, :] = image_tensor[:, :, :]
        results = results.view(-1, 120, 120)
    except Exception as e:
        print(f'due to error: {e}\nfailed to load video {video_path}')
    cap.release()
    return results


def check_ratio(path):
    with open(path) as f:
        content = f.read()
    return bool(content and float(content) >= 0.9)


class VideoDataset(Dataset):
    def __init__(self, path: str):
        super().__init__()

        self._dataset = []
        dirs = os.listdir(path)
        for dir in tqdm(dirs, desc='Load dataset'):
            all_mp4 = sorted(glob(os.path.join(path, dir, '*.mp4')))
            all_txt = sorted(glob(os.path.join(path, dir, '*.txt')))
            all_ratio = sorted(glob(os.path.join(path, dir, '*.ratio')))

            for mp4, txt, ratio in zip(all_mp4, all_txt, all_ratio):
                if not check_ratio(ratio):
                    continue

                with open(txt, 'r') as f:
                    try:
                        # remove `Text: `
                        first_line = f.readline().split()[1:]
                        # remove words in {}
                        first_line = [word for word in first_line if '{' not in word]
                        first_line = ' '.join(first_line)
                        chars = [CHAR_SET.index(i) for i in first_line]
                        # add eos to the end.
                        chars.append(CHAR_SET.index('<eos>'))
                        chars = torch.Tensor(chars)
                    except Exception as e:
                        print(e, txt)
                        continue
                self._dataset.append((mp4, chars))
        print(f'load {len(self._dataset)} samples')
    
    def __len__(self):
        return len(self._dataset)
    
    def __getitem__(self, index: int):
        video_path, chars = self._dataset[index]
        return load_video(video_path), chars
