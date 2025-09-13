import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class retroBERTdataset(Dataset):

    def __init__(
            self,
            data_dir,
            label_file,
            is_train=None,
            is_test=None,
            args=None,
        ):
        
        super().__init__()
        self.is_train = is_train
        self.is_test = is_test
        self.max_seq_length = args.max_seq_length
        self.step_size = args.max_seq_length // 2

        # Load data and prepare sequences
        self.data_list, self.label_list = self.load_data_labels(data_dir, label_file)
        self.source_seq, self.source_mask, self.target = self.generate_sequences(self.data_list, self.label_list)

    def load_data_labels(self, data_dir, label_file):
        all_data = []
        all_labels = []

        # Load labels from Excel or CSV file
        if label_file.endswith('.csv'):
            label_df = pd.read_csv(label_file, usecols=["file_name", "label"])
        else:
            label_df = pd.read_excel(label_file, usecols=["file_name", "label"])
        label_mapping = label_df.set_index('file_name')['label'].to_dict()

        # Map labels to binary format
        for key in label_mapping:
            if label_mapping[key] == "resilient":
                label_mapping[key] = 1
            elif label_mapping[key] == "susceptible":
                label_mapping[key] = 0

        # Origin correction values
        origin_x_values = [-0.1164, -0.0182, -0.0645, -0.0937, -0.2240]  
        origin_y_values = [-0.2297, -0.3475, -0.3701, -0.2918, -0.1989]

        # Collect all data first as tensors
        for filename in os.listdir(data_dir):
            if filename.endswith(".csv") and filename[:-4] in label_mapping:

                set_number = int(filename[3]) -1
                origin_x = origin_x_values[set_number]
                origin_y = origin_y_values[set_number]

                filepath = os.path.join(data_dir, filename)
                df = pd.read_csv(filepath)
                data_array = df.to_numpy(dtype=np.float32)

                # exclude the tail end coordinates
                data_array = data_array[:, :-3]

                # apply correction
                for i in range(8):
                    data_array[:, i*3] -= origin_x
                    data_array[:, i*3+1] -= origin_y

                tail_base = data_array[:, 6:9]
                body = data_array[:, 9:12]

                spine_length = np.linalg.norm(body - tail_base, axis=1)
                spine_length = spine_length.reshape(-1, 1)
                data_array /= spine_length

                tensor = torch.tensor(data_array, dtype=torch.float32)
                all_data.append(tensor)
                all_labels.append(label_mapping[filename[:-4]])

        return all_data, all_labels

    def generate_sequences(self, data_list, label_list):
        sequences = []
        masks = []
        labels = []

        if self.is_train:
            for data, label in zip(data_list, label_list):
                if label == 0:
                    unfolded_data = data.unfold(0, self.max_seq_length, self.step_size)
                    for i in range(unfolded_data.size(0)):
                        seq = unfolded_data[i].permute(1,0)
                        mask = torch.ones(self.max_seq_length+1)
                        sequences.append(seq)
                        masks.append(mask.long())
                        labels.append(label)
                else:
                    total_frames = data.shape[0]
                    num_seq = total_frames // self.max_seq_length

                    for i in range(num_seq):
                        start_index = i * self.max_seq_length
                        end_index = start_index + self.max_seq_length
                        seq = data[start_index:end_index]
                        
                        if seq.size(0) < self.max_seq_length:
                            pad_size = self.max_seq_length - seq.size(0)
                            seq = torch.cat([seq, torch.zeros(pad_size, seq.size(1), dtype=torch.float32)], dim=0)
                            mask = torch.cat([torch.ones(seq.size(0)), torch.zeros(pad_size)])  # attention_mask
                        else:
                            mask = torch.ones(self.max_seq_length+1)  # attention_mask
                        
                        sequences.append(seq)
                        masks.append(mask.long())
                        labels.append(label) 

            sequences = torch.stack(sequences)
            masks = torch.stack(masks)
            labels = torch.tensor(labels)

            return sequences, masks, labels
        
        else:
            for data, label in zip(data_list, label_list):
                total_frames = data.shape[0]
                num_seq = total_frames // self.max_seq_length

                for i in range(num_seq):
                    start_index = i * self.max_seq_length
                    end_index = start_index + self.max_seq_length
                    seq = data[start_index:end_index]
                    
                    if seq.size(0) < self.max_seq_length:
                        pad_size = self.max_seq_length - seq.size(0)
                        seq = torch.cat([seq, torch.zeros(pad_size, seq.size(1), dtype=torch.float32)], dim=0)
                        mask = torch.cat([torch.ones(seq.size(0)), torch.zeros(pad_size)])  # attention_mask
                    else:
                        mask = torch.ones(self.max_seq_length+1)  # attention_mask
                    
                    sequences.append(seq)
                    masks.append(mask.long())
                    labels.append(label)

            sequences = torch.stack(sequences)
            masks = torch.stack(masks)
            labels = torch.tensor(labels)

            return sequences, masks, labels

    def generate_test_sequences(self, data):
        sequences = []
        masks = []
        total_frames = data.shape[0]
        num_seq = total_frames // self.max_seq_length

        for i in range(num_seq):
            start_index = i * self.max_seq_length
            end_index = start_index + self.max_seq_length
            seq = data[start_index:end_index]
            
            if seq.size(0) < self.max_seq_length:
                pad_size = self.max_seq_length - seq.size(0)
                seq = torch.cat([seq, torch.zeros(pad_size, seq.size(1), dtype=torch.float32)], dim=0)
                mask = torch.cat([torch.ones(seq.size(0)), torch.zeros(pad_size)])  # attention_mask
            else:
                mask = torch.ones(self.max_seq_length+1)  # attention_mask
            
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
