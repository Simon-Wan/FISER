import os
import json

import torch
from torch.utils.data import DataLoader

from dataloaders.data_utils import Batch


def get_dataloaders(data_dir, train_files, valid_files, src_tokenizer, device, mid_tokenizers, batch_size):
    train_dataset = Dataset(data_dir, train_files, src_tokenizer, device, mid_tokenizers)
    valid_dataset = Dataset(data_dir, valid_files, src_tokenizer, device, mid_tokenizers)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=lambda x: make_batch(x, device),
                                  drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, 1, shuffle=False, collate_fn=lambda x: make_batch(x, device),
                                  drop_last=True)
    return train_dataloader, valid_dataloader


def get_eval_dataloader(data_dir, valid_files, src_tokenizer, device, mid_tokenizers):
    valid_dataset = Dataset(data_dir, valid_files, src_tokenizer, device, mid_tokenizers)
    valid_dataloader = DataLoader(valid_dataset, 1, shuffle=False, collate_fn=lambda x: make_batch(x, device),
                                  drop_last=True)
    return valid_dataloader


class Dataset(object):
    def __init__(self, data_dir, json_files, src_tokenizer, device, mid_tokenizers=None):
        self.data_dir = data_dir
        self.json_files = json_files
        self.src_tokenizer = src_tokenizer
        self.device = device
        self.mid_tokenizers = mid_tokenizers
        self.id2episode_map = self._load_episodes()
        self.episode_processed_cache = dict()

    def __len__(self):
        return len(self.id2episode_map)

    def __getitem__(self, idx):
        episode_index, within_episode_index = self.id2episode_map[idx]
        if episode_index in self.episode_processed_cache:
            episode = self.episode_processed_cache[episode_index]
        else:
            episode = self._process_episode(episode_index)
            self.episode_processed_cache[episode_index] = episode
        return episode[within_episode_index]

    def _load_episodes(self):
        # define mappings from (idx) to (episode,idx)
        id2episode_map = dict()
        idx = 0
        for episode_index, file in enumerate(self.json_files):
            with open(os.path.join(self.data_dir, file), 'r') as f:
                json_str = json.load(f)
                length = len(json_str['demo_actions'])
                for within_episode_index in range(length):
                    id2episode_map[idx] = (episode_index, within_episode_index)
                    idx += 1
        return id2episode_map

    def _process_episode(self, episode_index):
        pt_filename = self.json_files[episode_index].replace('json', 'pt')
        episode = torch.load(os.path.join(self.data_dir, '../preprocessed', pt_filename))
        return episode


def make_batch(batch_list, device):
    batch_size = len(batch_list)

    graphs = torch.stack([b.src[0] for b in batch_list]).to(device)
    initial_graphs = torch.stack([b.src[4] for b in batch_list]).to(device)
    trajectories = torch.stack([b.src[1] for b in batch_list]).to(device)
    instructions = torch.stack([b.src[2] for b in batch_list]).to(device)
    effects = torch.stack([b.src[3] for b in batch_list]).to(device)
    actions = torch.stack([b.tgt[0] for b in batch_list]).to(device)
    arg1s = torch.stack([b.tgt[1] for b in batch_list]).to(device)
    arg2s = torch.stack([b.tgt[2] for b in batch_list]).to(device)
    solutions = torch.stack([b.solutions for b in batch_list]).to(device)
    target_obj_indices = torch.stack([b.target_obj_indices for b in batch_list]).to(device)

    choices = [b.choices for b in batch_list]
    choices_in_text = [b.choices_in_text for b in batch_list]
    json_indices = [b.json_indices for b in batch_list]
    object_names = [b.object_names for b in batch_list]

    subgoals = torch.stack([b.subgoals for b in batch_list]).to(device)
    mid_q = torch.stack([b.qsvo[0] for b in batch_list]).to(device)
    mid_s = torch.stack([b.qsvo[1] for b in batch_list]).to(device)
    mid_v = torch.stack([b.qsvo[2] for b in batch_list]).to(device)
    mid_o = torch.stack([b.qsvo[3] for b in batch_list]).to(device)

    mask0 = torch.stack([b.src_mask[0] for b in batch_list]).to(device)
    mask1 = torch.stack([b.src_mask[1] for b in batch_list]).to(device)
    mask2 = torch.stack([b.src_mask[2] for b in batch_list]).to(device)
    mask3 = torch.stack([b.src_mask[3] for b in batch_list]).to(device)

    replace_indices = [b.replace_indices.to(device) for b in batch_list]
    object_indices = [b.object_indices.to(device) for b in batch_list]

    tensor_batch = Batch()
    tensor_batch.size = batch_size
    tensor_batch.src = (graphs, trajectories, instructions, effects, initial_graphs)
    tensor_batch.tgt = (actions, arg1s, arg2s)
    tensor_batch.choices = choices                      # not tensor
    tensor_batch.choices_in_text = choices_in_text      # not tensor
    tensor_batch.json_indices = json_indices            # not tensor
    tensor_batch.object_names = object_names            # not tensor
    tensor_batch.subgoals = subgoals
    tensor_batch.qsvo = (mid_q, mid_s, mid_v, mid_o)
    tensor_batch.src_mask = (mask0, mask1, mask2, mask3)
    tensor_batch.solutions = solutions
    tensor_batch.target_obj_indices = target_obj_indices

    tensor_batch.replace_indices = replace_indices
    tensor_batch.object_indices = object_indices
    return tensor_batch
