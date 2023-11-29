import os
import numpy as np

import torch
from tqdm import tqdm
from torch.utils.data import Dataset

class MeshSplatDataset(Dataset):
    def __init__(self, root, split='train', num_points=1024, block_size=1.0, sample_rate=1.0, transform=None, device='cuda:0'):
        super().__init__()
        self.root = root
        self.split = split
        self.num_points = num_points
        self.block_size = block_size
        self.sample_rate = sample_rate
        self.transform = transform
        self.device = device
        self.points = []
        self.splats = []
        

        if self.split == 'train':
            self.files = [fil.split('.')[0] for fil in os.listdir(os.path.join(self.root, 'train')) if not fil.endswith('_splat.npy')]
        elif self.split == 'test':
            self.files = [fil.split('.')[0] for fil in os.listdir(os.path.join(self.root, 'test')) if not fil.endswith('_splat.npy')]
        else:
            raise ValueError('Unsupported split: %s' % self.split)
        
        for filename in self.files:
            points = np.load(os.path.join(self.root, self.split, filename+'.npy'))
            splats = np.load(os.path.join(self.root, self.split, filename+'_splat.npy'))

            max_pts = np.amax(points[:, :3], axis=0)
            current_points = np.zeros((points.shape[0], 9))  # num_point * 9
            current_points[:, :6] = points
            current_points[:, 6:] = points[:, :3]/max_pts
            current_points[:, 3:6] /= 255.0
            
            self.points.append(current_points.transpose(1, 0))
            self.splats.append(splats.transpose(1, 0))
        
        print('The size of %s data is %d' % (self.split, len(self.points)))
    
    def __getitem__(self, index):
        points = torch.Tensor(self.points[index]).to(self.device)
        splats = torch.Tensor(self.splats[index]).to(self.device)
        if self.transform is not None:
            points, splats = self.transform(points, splats)
        return points, splats
    
    def __len__(self):
        return len(self.points)