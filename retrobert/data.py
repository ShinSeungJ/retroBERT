"""Dataset: pose CSVs -> padded, masked sequence tensors."""

import os
import math
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils.data_utils import normalize_spine

class retroBERTdataset(Dataset):

    def __init__(
            self,
            data_dir,
            label_file,
            is_train=None,
            is_test=None,
            shuffle_targets=False,
            args=None,
        ):

        super().__init__()
        self.is_train = is_train
        self.is_test = is_test
        self.args = args
        self.num_frames = getattr(args, 'num_frames', args.max_seq_length - 1)
        self.step_size = self.num_frames // 2

        # Negative controls. shuffle_targets permutes this split's animal<->label
        # links; shuffle_sequences permutes the frame order inside every window.
        self.shuffle_targets = shuffle_targets
        self.shuffle_sequences = getattr(args, 'shuffle', 'none') == 'sequences'
        self.shuffle_seed = getattr(args, 'shuffle_seed', 0)

        self.data_list, self.label_list = self.load_data_labels(data_dir, label_file)
        self.source_seq, self.source_mask, self.target = self.generate_sequences(self.data_list, self.label_list)
        # Features per frame, read off the data rather than declared up front.
        self.input_dim = int(self.source_seq[0].shape[-1])

    def load_data_labels(self, data_dir, label_file):
        all_data = []
        all_labels = []

        label_df = pd.read_excel(label_file, usecols=["name", "group"])
        label_mapping = label_df.set_index('name')['group'].to_dict()

        for key in label_mapping:
            if label_mapping[key] == "resilient":
                label_mapping[key] = 1
            elif label_mapping[key] == "susceptible":
                label_mapping[key] = 0

        for filename in os.listdir(data_dir):
            if filename.endswith(".csv") and filename[:-4] in label_mapping:

                filepath = os.path.join(data_dir, filename)
                df = pd.read_csv(filepath)
                data_array = df.to_numpy(dtype=np.float32)
                data_array = normalize_spine(data_array, self.args.spine_scale)

                tensor = torch.tensor(data_array, dtype=torch.float32)
                all_data.append(tensor)
                all_labels.append(label_mapping[filename[:-4]])

        if self.shuffle_targets:
            np.random.RandomState(self.shuffle_seed).shuffle(all_labels)

        return all_data, all_labels

    def generate_sequences(self, data_list, label_list):
        sequences = []
        masks = []
        labels = []

        if self.is_train:
            seq_counter = 0
            for data, label in zip(data_list, label_list):
                unfolded_data = data.unfold(0, self.num_frames, self.step_size)
                num_windows = unfolded_data.size(0)
                for i in range(num_windows):
                    seq = unfolded_data[i].permute(1, 0)
                    if self.shuffle_sequences:
                        perm = np.random.RandomState(self.shuffle_seed + seq_counter).permutation(self.num_frames)
                        seq = seq[perm]
                    mask = torch.ones(self.num_frames + 1)
                    sequences.append(seq)
                    masks.append(mask.long())
                    labels.append(label)
                    seq_counter += 1

                tail_start = (num_windows - 1) * self.step_size + self.num_frames if num_windows > 0 else 0
                if tail_start < data.size(0):
                    tail = data[tail_start:]
                    original_size = tail.size(0)
                    pad_size = self.num_frames - original_size
                    tail_padded = torch.cat([tail, torch.zeros(pad_size, tail.size(1), dtype=torch.float32)], dim=0)
                    if self.shuffle_sequences and original_size > 1:
                        perm = np.random.RandomState(self.shuffle_seed + seq_counter).permutation(original_size)
                        tail_padded[:original_size] = tail_padded[:original_size][perm]
                    mask = torch.cat([torch.ones(1 + original_size), torch.zeros(pad_size)])
                    sequences.append(tail_padded)
                    masks.append(mask.long())
                    labels.append(label)
                    seq_counter += 1

            sequences = torch.stack(sequences)
            masks = torch.stack(masks)
            labels = torch.tensor(labels)

            return sequences, masks, labels

        else:
            seq_counter = 0
            for data, label in zip(data_list, label_list):
                total_frames = data.shape[0]
                num_seq = math.ceil(total_frames / self.num_frames)

                for i in range(num_seq):
                    start_index = i * self.num_frames
                    end_index = start_index + self.num_frames
                    seq = data[start_index:end_index]

                    if seq.size(0) < self.num_frames:
                        original_size = seq.size(0)
                        pad_size = self.num_frames - original_size
                        seq = torch.cat([seq, torch.zeros(pad_size, seq.size(1), dtype=torch.float32)], dim=0)
                        if self.shuffle_sequences and original_size > 1:
                            perm = np.random.RandomState(self.shuffle_seed + seq_counter).permutation(original_size)
                            seq[:original_size] = seq[:original_size][perm]
                        mask = torch.cat([torch.ones(1 + original_size), torch.zeros(pad_size)])
                    else:
                        if self.shuffle_sequences:
                            perm = np.random.RandomState(self.shuffle_seed + seq_counter).permutation(self.num_frames)
                            seq = seq[perm]
                        mask = torch.ones(self.num_frames + 1)

                    sequences.append(seq)
                    masks.append(mask.long())
                    labels.append(label)
                    seq_counter += 1

            sequences = torch.stack(sequences)
            masks = torch.stack(masks)
            labels = torch.tensor(labels)

            return sequences, masks, labels

    def generate_test_sequences(self, data):
        sequences = []
        masks = []
        total_frames = data.shape[0]
        num_seq = math.ceil(total_frames / self.num_frames)

        for i in range(num_seq):
            start_index = i * self.num_frames
            end_index = start_index + self.num_frames
            seq = data[start_index:end_index]

            if seq.size(0) < self.num_frames:
                original_size = seq.size(0)
                pad_size = self.num_frames - original_size
                seq = torch.cat([seq, torch.zeros(pad_size, seq.size(1), dtype=torch.float32)], dim=0)
                if self.shuffle_sequences and original_size > 1:
                    perm = np.random.RandomState(self.shuffle_seed + i).permutation(original_size)
                    seq[:original_size] = seq[:original_size][perm]
                mask = torch.cat([torch.ones(1 + original_size), torch.zeros(pad_size)])
            else:
                if self.shuffle_sequences:
                    perm = np.random.RandomState(self.shuffle_seed + i).permutation(self.num_frames)
                    seq = seq[perm]
                mask = torch.ones(self.num_frames + 1)

            sequences.append(seq)
            masks.append(mask.long())

        return torch.stack(sequences), torch.stack(masks)

    def __len__(self):
        return len(self.source_seq)

    def __getitem__(self, index):
        source_seq = self.source_seq[index]
        source_mask = self.source_mask[index]
        target = self.target[index]
        return {"input": source_seq,
                "mask": source_mask,
                "target": target}

    def collate_fn(self, batch):
        inputs = torch.stack([x['input'] for x in batch])
        masks = torch.stack([x['mask'] for x in batch])
        targets = torch.stack([x['target'] for x in batch])
        return {"input": inputs, "mask": masks, "target": targets}
