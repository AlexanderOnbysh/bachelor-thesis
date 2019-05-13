from torch import nn
import torch
from torch.utils.checkpoint import checkpoint_sequential

from constants import CHAR_SET


class VGG(nn.Module):
    """VGG-M: https://www.robots.ox.ac.uk/~vgg/publications/2016/Chung16/chung16.pdf.

    Modified the architecture a little. In the paper they consider 5 frames as a unit.
    Here, we start with a single frame. May need to tune it in the future.

    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(5, 96, (7, 7), (2, 2)),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True),
            nn.Conv2d(96, 256, (5, 5), (2, 2), (1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((3, 3),(2, 2),(0, 0), ceil_mode=True)
        )
        self.fc = nn.Linear(4608, 512)

    def forward(self, x):
        """Expect x with shape (# of timestep, 5, 120, 120)"""
        return self.fc(checkpoint_sequential(self.encoder, len(self.encoder), x).view(x.size(0), -1))

    def forward_(self, x):
        """Expect x with shape (batch, seq_len, 120, 120)"""
        batch_size, seq_len = x.shape[:2]
        # batch_size * seq_len, 3, 3, 512
        x = self.features(x.view(-1, 1, 120, 120))
        # flatten to (_, 4608)
        x = x.view(batch_size * seq_len, -1)
        # (_, 512) mapping of frames to classes
        x = self.classifier(x)
        # view to (batch_size, seq_len, features)
        x = x.view(batch_size, seq_len, -1)
        return x


class Watch(nn.Module):
    """Feed video to VGG and LSTM."""
    def __init__(self):
        super().__init__()
        self.encoder = VGG()
        self.lstm = nn.LSTM(512, 512, num_layers=3, batch_first=True)
    
    def forward_(self, x):
        output_from_vgg = self.vgg(x).view(1, -1, 512)  # (# of timestep, 512)
        output_from_vgg_lstm, states_from_vgg_lstm = self.lstm(output_from_vgg)

        # output_from_vgg_lstm: (_, 1, 512)
        # states_from_vgg_lstm[0]: (3, _, 512)
        return output_from_vgg_lstm, states_from_vgg_lstm[0]

    def forward(self, x, length):
        length = x.size(1)
        # assert len(size) == 4, 'video input size is wrong'
        # assert size[2:] == [120, 120], 'image size should 120 * 120'
        x = self.encoder(x.view(-1, 1, 120, 120)).view(-1, length, 512)
        length, perm_idx = length.sort(0, descending=True)

        # for i, tensor in enumerate(x):
        #     tensor[length[i]:, :] = torch.tensor(0)

        x = x[perm_idx].squeeze(1)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths=length.view(-1).int(), batch_first=True)
        # self.lstm.flatten_parameters()
        outputs, states = self.lstm(x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True, padding_value=0)
        outputs = outputs[perm_idx.sort()[1]]
        return outputs, states[0]


class Attention(nn.Module):
    """Reference: https://machinelearningmastery.com/how-does-attention-work-in-encoder-decoder-recurrent-neural-networks/
    The attention layer.

    """
    def __init__(self, hidden_size, annotation_size):
        super(Attention, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(hidden_size + annotation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, prev_hidden_state, annotations):
        batch_size, sequence_length = annotations.size(0), annotations.size(1)
        prev_hidden_state = prev_hidden_state.repeat(sequence_length, 1, 1).transpose(0, 1)
        concatenated = torch.cat([prev_hidden_state, annotations], dim=2)
        alpha = torch.nn.functional.softmax(self.dense(concatenated).squeeze(2)).unsqueeze(1)

        return alpha.bmm(annotations)


class Spell(nn.Module):
    """Reference: https://www.robots.ox.ac.uk/~vgg/publications/2017/Chung17/chung17.pdf"""
    def __init__(self):
        super().__init__()
        self.hidden_size = 512
        self.output_size = len(CHAR_SET)
        self.num_layers = 3

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size * 2, self.hidden_size, self.num_layers, batch_first=True)
        self.attention = Attention(self.hidden_size, self.hidden_size)

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_size)
        )

    def forward(self, spell_input, hidden_state, spell_state, watcher_outputs, context):
        spell_input = self.embedding(spell_input)
        concatenated = torch.cat([spell_input, context], dim=2)
        output, (hidden_state, spell_state) = self.lstm(concatenated, (hidden_state, spell_state))
        context = self.attention(hidden_state[-1], watcher_outputs)
        output = self.mlp(torch.cat([output, context], dim=2).squeeze(1)).unsqueeze(1)

        return output, hidden_state, spell_state, context
