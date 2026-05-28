import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, stable, label, window_size):
        self.data = data
        self.stable = stable
        self.label = label
        self.window_size = window_size

    def __getitem__(self, index):
        data = self.data[index: index + self.window_size, :]
        stable = self.stable[index: index + self.window_size, :]
        label = self.label[index: index + self.window_size, :]

        return data, stable, label

    def __len__(self):
        return len(self.data) - self.window_size + 1


class NonOverlapDataset(torch.utils.data.Dataset):
    """Dataset with non-overlapping windows for evaluation."""
    def __init__(self, data, stable, label, window_size):
        self.data = data
        self.stable = stable
        self.label = label
        self.window_size = window_size
        
        # Number of non-overlapping windows.
        self.num_samples = len(self.data) // self.window_size

    def __getitem__(self, index):
        start_idx = index * self.window_size
        end_idx = start_idx + self.window_size
        
        data = self.data[start_idx:end_idx, :]
        stable = self.stable[start_idx:end_idx, :]
        label = self.label[start_idx:end_idx, :]

        return data, stable, label

    def __len__(self):
        return self.num_samples
