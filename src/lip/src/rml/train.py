import os
from glob import glob

import cv2
import editdistance
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from constants import (CHAR_SET, LEARNING_RATE, LEARNING_RATE_GAMMA, LEARNING_RATE_STEP)
from dataset import VideoDataset
from model import Spell, Watch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def run(dataset_path):
    os.makedirs('checkpoints', exist_ok=True)
    writer = SummaryWriter()

    watcher = Watch().to('cuda' if torch.cuda.is_available() else 'cpu')
    speller = Spell().to('cuda' if torch.cuda.is_available() else 'cpu')


    # Applying learning rate decay as we observed loss diverge.
    # https://discuss.pytorch.org/t/how-to-use-torch-optim-lr-scheduler-exponentiallr/12444/6
    watch_optimizer = optim.Adam(watcher.parameters(), lr=LEARNING_RATE)
    spell_optimizer = optim.Adam(speller.parameters(), lr=LEARNING_RATE)
    watch_scheduler = optim.lr_scheduler.StepLR(watch_optimizer, step_size=LEARNING_RATE_STEP, gamma=LEARNING_RATE_GAMMA)
    spell_scheduler = optim.lr_scheduler.StepLR(spell_optimizer, step_size=LEARNING_RATE_STEP, gamma=LEARNING_RATE_GAMMA)

    dataset = VideoDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    criterion = nn.CrossEntropyLoss()

    losses = []
    for epoch in range(1, 100):
        watch_scheduler.step()
        spell_scheduler.step()

        watcher = watcher.train()
        speller = speller.train()

        for i, (x, chars) in tqdm(enumerate(dataloader, 1), total=len(dataloader), desc=f'Epoch: {epoch}'):
            chars = chars[0]
            loss = 0
            watch_optimizer.zero_grad()
            spell_optimizer.zero_grad()

            x = x.to(device)
            chars = chars.to(device)
            chars_len = chars.size(0)

            output_from_vgg_lstm, states_from_vgg_lstm = watcher(x)

            spell_input = torch.tensor([[CHAR_SET.index('<sos>')]]).repeat(output_from_vgg_lstm.size(0), 1).to(device)
            spell_hidden = states_from_vgg_lstm
            spell_state = torch.zeros_like(spell_hidden).to(device)
            context = torch.zeros(output_from_vgg_lstm.size(0), 1, spell_hidden.size(2)).to(device)

            for idx in range(chars_len):
                spell_output, spell_hidden, spell_state, context = speller(spell_input, spell_hidden, spell_state, output_from_vgg_lstm, context)
                _, topi = spell_output.topk(1, dim=2)
                spell_input = chars[idx].long().view(1, 1)
                loss += criterion(spell_output.squeeze(1), chars[idx].long().view(1))

            loss = loss.to('cuda' if torch.cuda.is_available() else 'cpu')
            loss.backward()
            watch_optimizer.step()
            spell_optimizer.step()

            norm_loss = float(loss / chars.size(0))
            losses.append(norm_loss)

            writer.add_scalar('train/loss', norm_loss, global_step=epoch * len(dataloader) + i)
            writer.add_scalar('train/lr-watcher', watch_scheduler.get_lr()[0], global_step=epoch * len(dataloader) + i)
            writer.add_scalar('train/lr-speller', spell_scheduler.get_lr()[0], global_step=epoch * len(dataloader) + i)

        watcher = watcher.eval()
        speller = speller.eval()

        torch.save({
            'watcher': watcher.state_dict(),
            'speller': speller.state_dict(),
            }, f'checkpoints/{epoch:03d}_{norm_loss}.pth')


    print(f'{losses}')

if __name__ == '__main__':
    run('/mnt/disk/alexon/pretrain-cropped')
