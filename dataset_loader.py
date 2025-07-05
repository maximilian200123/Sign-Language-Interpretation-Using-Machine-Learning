#ORIGINAL CODE

# import os
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from torch.nn.utils.rnn import pad_sequence
# from glob import glob

# class LandmarksDataset(Dataset):
#     def __init__(self, root_dir, mode='train', max_glosses=None):
#         assert mode in ['train', 'val'], "mode must be 'train' or 'val'"
#         self.samples = []
#         self.labels = []
#         self.label_to_idx = {}
#         self.idx_to_label = {}

#         glosses = sorted(os.listdir(root_dir))
#         glosses = [g for g in glosses if os.path.isdir(os.path.join(root_dir, g))]

#         if max_glosses is not None:
#             glosses = glosses[:max_glosses]

#         for idx, gloss in enumerate(glosses):
#             gloss_path = os.path.join(root_dir, gloss)
#             if not os.path.isdir(gloss_path):
#                 continue

#             self.label_to_idx[gloss] = idx
#             self.idx_to_label[idx] = gloss

#             mode_path = os.path.join(gloss_path, mode)
#             if not os.path.exists(mode_path):
#                 continue

#             for npy_file in glob(os.path.join(mode_path, "*.npy")):
#                 self.samples.append(npy_file)
#                 self.labels.append(idx)

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         data = np.load(self.samples[idx])
#         if data.ndim == 3:
#             data = data.reshape(data.shape[0], -1)  # Flatten if needed

#         length = data.shape[0]  # original sequence length
#         x = torch.tensor(data, dtype=torch.float32)
#         y = torch.tensor(self.labels[idx], dtype=torch.long)

#         return x, length, y  # Return variable-length data


# def collate_fn(batch):
#     # Sort the batch by sequence length in descending order
#     batch.sort(key=lambda x: x[1], reverse=True)
#     sequences, lengths, labels = zip(*batch)

#     # Pad sequences to the max length in the batch
#     padded_sequences = pad_sequence(sequences, batch_first=True)

#     lengths = torch.tensor(lengths, dtype=torch.long)
#     labels = torch.tensor(labels, dtype=torch.long)

#     return padded_sequences, lengths, labels

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob

class LandmarksDataset(Dataset):
    def __init__(self, root_dir, mode="train", max_glosses=None):
        self.samples = []
        self.labels = []
        self.label_to_idx = {}
        self.idx_to_label = {}

        glosses = sorted(os.listdir(root_dir))
        if max_glosses:
            glosses = glosses[:max_glosses]

        for idx, gloss in enumerate(glosses):
            gloss_path = os.path.join(root_dir, gloss)
            mode_path = os.path.join(gloss_path, mode)
            if not os.path.isdir(mode_path):
                continue

            self.label_to_idx[gloss] = idx
            self.idx_to_label[idx] = gloss

            for sequence_dir in glob(os.path.join(mode_path, "*")):
                if not os.path.isdir(sequence_dir):
                    continue
                frames = sorted(glob(os.path.join(sequence_dir, "*.npy")))
                if len(frames) < 3:
                    continue  # Skip too-short sequences
                self.samples.append((frames, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]
        frames = [np.load(f) for f in frame_paths]
        data = np.stack(frames)  
        data = self.normalize_sample(data)      
        data = data.reshape(data.shape[0], -1)  
        x = torch.tensor(data, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)
        return x, x.shape[0], y
    
    def normalize_sample(self, sample):
        x = sample[:, :, 0]
        y = sample[:, :, 1]
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        scale = np.sqrt(np.var(x_centered) + np.var(y_centered)) + 1e-6
        sample[:, :, 0] = x_centered / scale
        sample[:, :, 1] = y_centered / scale
        return sample


def collate_fn(batch):
    batch.sort(key=lambda x: x[1], reverse=True)
    sequences, lengths, labels = zip(*batch)
    padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    return padded_sequences, torch.tensor(lengths), torch.tensor(labels)

