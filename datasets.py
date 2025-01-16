import math
import torch
import numpy as np
from torch.utils.data import Dataset

import settings


class RotatedUnitCircleDataset(Dataset):
    def __init__(self, num_samples=20, shift_degree=45):
        self.num_samples = num_samples
        self.shift_radians = math.radians(shift_degree)
        angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
        angles_shifted = angles + self.shift_radians
        self.data_points = np.vstack((np.cos(angles_shifted), np.sin(angles_shifted))).T
        self.labels = np.array([i % 2 for i in range(num_samples)])
        self.labels = np.expand_dims(self.labels, -1).astype(np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.from_numpy(self.data_points[idx]).float().to(settings.device), torch.tensor(self.labels[idx],
                                                                                             dtype=torch.float).to(
            settings.device)


class RotatedUnitCircleDataset__(Dataset):
    def __init__(self, num_samples=20, shift_degree=45):
        self.num_samples = num_samples
        angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
        # self.shift_radians = np.array([math.radians(shift_degree + random.gauss(sigma=45)) for _ in range(len(angles))])
        self.shift_radians = math.radians(shift_degree)
        angles_shifted = angles + self.shift_radians
        # radii = np.array([random.gauss(0.8, 0.1) if i % 2 == 0 else random.gauss(1.2, 0.1) for i in range(num_samples)])
        # radii = np.array([0.8 if i % 2 == 0 else 1.2 for i in range(num_samples)])
        radii = np.array([0.999 if i % 2 == 0 else 1.001 for i in range(num_samples)])
        self.data_points = np.vstack((radii * np.cos(angles_shifted), radii * np.sin(angles_shifted))).T
        self.labels = np.array([i % 2 for i in range(num_samples)])
        self.labels = np.expand_dims(self.labels, -1).astype(np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.from_numpy(self.data_points[idx]).float().to(settings.device), torch.tensor(self.labels[idx],
                                                                                             dtype=torch.long).to(
            settings.device)
