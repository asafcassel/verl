import torch
import numpy as np
from torch.utils.data import Sampler

class HedgeSampler(Sampler):
    def __init__(self, data_source, batch_size, generator, init_type, replacement=True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.generator = generator
        self.init_type = init_type
        self.replacement = replacement

        if (not replacement) and batch_size > len(data_source):
            raise f"{batch_size=} cannot allow sampling without replacement from data size f{len(data_source)}"

        # Extract difficulty levels from dataset
        self.difficulties = torch.tensor([data_source[i]['difficulty'] / 100.0 for i in range(len(data_source))])
        self.difficulties_squared = torch.square(self.difficulties)

        if init_type == 'uniform':
            self.logits = torch.zeros(len(data_source))
        elif init_type == 'softmax':
            self.logits = self.difficulties.detach().clone()
        else:
            raise f"{init_type=} is not a valid choice"
        
        self.probs = torch.nn.functional.softmax(self.logits, dim=0)
        self.cur_samples = []
        
    def __iter__(self):
        num_samples = len(self.data_source)
        cur_num_samples = 0
        self.cur_samples = []
        
        while cur_num_samples < num_samples:
            selected_indices = torch.multinomial(self.probs, self.batch_size, replacement=self.replacement, generator=self.generator)
            self.cur_samples = [int(i) for i in selected_indices]
            cur_num_samples += len(self.cur_samples)
            yield [int(i) for i in selected_indices]
    
    def update_sampling_probabilities(self, rewards, eta, gamma):
        """Update the target difficulty dynamically based on model feedback."""
        hedge_losses = rewards
        assert len(self.cur_samples) == self.batch_size, f"{len(self.cur_samples)}, {self.batch_size}"
        indices = torch.tensor([self.cur_samples, [i for i in range(self.batch_size)]])
        sample_losses = torch.sparse_coo_tensor(indices, hedge_losses, size=(len(self.data_source),self.batch_size)).sum(-1).to_dense()
        if gamma > 0:
            sample_losses /= (self.probs + gamma)
        self.logits -= (eta * sample_losses)
        self.probs = torch.nn.functional.softmax(self.logits, dim=0)
        self.cur_samples = []
        
        # Return difficulty average and std
        difficulty_mean = torch.dot(self.difficulties, self.probs)
        difficulty_second_moment = torch.dot(self.difficulties_squared, self.probs)
        difficulty_std = torch.sqrt(torch.maximum(difficulty_second_moment - torch.square(difficulty_mean), torch.tensor([0.0])))
        sample_metrics = {
            'critic/difficulty_mean': difficulty_mean.item(),
            'critic/difficulty_std': difficulty_std.item(),
            'critic/difficulty_sample_mean': torch.mean(self.difficulties[self.cur_samples]).item(),
            'critic/difficulty_sample_std': torch.std(self.difficulties[self.cur_samples]).item(),
        }
        return sample_metrics

    def __len__(self):
        return len(self.data_source) // self.batch_size
