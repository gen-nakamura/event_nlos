import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from utils import create_data_check
from PIL import Image
import numpy as np
import os
import torch

class CustomDataset(Dataset):

    def __init__(self, input_dir, target_dir, title, num_bins):
        self.dir_input = input_dir
        self.dir_target = target_dir
        self.title = title
        self.num_bins = num_bins
        # for user in range(subject, subject+1):  # Looping over all users
        #     for eye in [0, 1]:
        #         user_dir = os.path.join(img_dir, f"user{user}", str(eye), 'binary_image')
        #         self.img_paths.extend(self._get_img_paths(user_dir))
        # self.transform = transforms.Compose([transforms.ToTensor()])


    def __getitem__(self, idx):
        input_path_high = self.dir_input + self.title + str(idx).zfill(5) + f"_{self.num_bins}.npz"
        event_voxel = np.load(input_path_high)['arr_0']

        target_path = self.dir_target + self.title + str(idx).zfill(5) + f"_{self.num_bins}.npz"
        labels = np.load(target_path)['arr_0']
        assert labels.shape == (150, 2), f"the shape is not what you expected: {labels.shape}"
        assert event_voxel.shape == (150, 480, 640), f"the shape is not what you expected: {event_voxel.shape}"
        # print("before normalization: ", labels)
        labels[:, 0] = (labels[:, 0]* -1 - 860) / 240
        labels[:, 1] = (labels[:, 1] - 650) / 300

        file_name = self.title + str(idx).zfill(5)
        print(f'creating a data of {file_name}')
        create_data_check(event_voxel, labels, file_name)
        return torch.FloatTensor(event_voxel), torch.FloatTensor(labels)

    def _get_img_paths(self, img_dir):
        img_dir = Path(img_dir)
        img_paths = [p for p in img_dir.iterdir() if p.suffix in [".jpg", ".jpeg", ".png", ".bmp"]]
        return img_paths

    def __len__(self):
        return len(self.img_paths)
