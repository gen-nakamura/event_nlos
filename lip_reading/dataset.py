import glob
import numpy as np
import os
from torch.utils.data import Dataset
import torch
from cvtransforms import *
from tqdm import tqdm  # tqdmをインポート
from torchvision.transforms import Resize, Compose

def events_to_voxel_grid_pytorch(events, num_bins, width, height, device):

    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    with torch.no_grad():

        events_torch = torch.from_numpy(events).float()
        events_torch = events_torch.to(device)

        voxel_grid = torch.zeros(num_bins, height, width, dtype=torch.float32, device=device)
        if events_torch.shape[0] == 0:
            return voxel_grid

        voxel_grid = voxel_grid.flatten()

        # normalize the event timestamps so that they lie between 0 and num_bins
        last_stamp = events_torch[-1, 0]
        first_stamp = events_torch[0, 0]
        deltaT = float(last_stamp - first_stamp)

        if deltaT == 0:
            deltaT = 1.0

        events_torch[:, 0] = (num_bins - 1) * (events_torch[:, 0] - first_stamp) / deltaT
        ts = events_torch[:, 0]
        xs = events_torch[:, 1].long()
        ys = events_torch[:, 2].long()
        pols = events_torch[:, 3].float()
        pols[pols == 0] = -1  # polarity should be +1 / -1

        tis = torch.floor(ts)
        tis_long = tis.long()
        dts = ts - tis
        vals_left = pols * (1.0 - dts.float())
        vals_right = pols * dts.float()

        valid_indices = tis < num_bins
        valid_indices &= tis >= 0
        voxel_grid.index_add_(dim=0,
                              index=xs[valid_indices] + ys[valid_indices]
                                    * width + tis_long[valid_indices] * width * height,
                              source=vals_left[valid_indices])

        valid_indices = (tis + 1) < num_bins
        valid_indices &= tis >= 0

        voxel_grid.index_add_(dim=0,
                              index=xs[valid_indices] + ys[valid_indices] * width
                                    + (tis_long[valid_indices] + 1) * width * height,
                              source=vals_right[valid_indices])

        voxel_grid = voxel_grid.view(num_bins, height, width)

    return voxel_grid

def events_to_voxel_all(events, frame_nums, seq_len, num_bins, width, height, device):
    voxel_len = min(seq_len, frame_nums) * num_bins # frame_numsで調節すればおk
    voxel_grid_all = np.zeros((num_bins * seq_len, 1, height, width))
    voxel_grid = events_to_voxel_grid_pytorch(events, voxel_len, width, height, device)
    voxel_grid = voxel_grid.unsqueeze(1).cpu().numpy()
    voxel_grid_all[:voxel_len] = voxel_grid
    return voxel_grid_all


class CustomDataset(Dataset):
    def __init__(self, input_dir, target_dir, title):
        self.dir_input = input_dir
        self.dir_target = target_dir
        self.title = title
        # for idx in range(10):
        #     input_path_high = self.dir_input + self.title + str(idx).zfill(5) + f"_{150}.npz"
        #     input_loaded_high = np.load(input_path_high)
        #     event_voxel_high = input_loaded_high['arr_0']

        #     input_path_low = self.dir_input + self.title + str(idx).zfill(5) + f"_{30}.npz"
        #     input_loaded_low = np.load(input_path_low)
        #     event_voxel_low = input_loaded_low['arr_0']

        #     target_path = self.dir_target + self.title + str(idx).zfill(5) + f"_{30}.npz"
        #     label = np.load(target_path)['arr_0']
        #     if event_voxel_high.shape != (150, 480, 640):
        #         print('high: ', event_voxel_high.shape)
        #         print()
        #     if event_voxel_low.shape != (30, 480, 640):
        #         print('low: ', event_voxel_low.shape)
        #         print()
        #     if label.shape != (30, 2):
        #         print('label: ', label.shape)
        #         print()


        # self.data_path = data_path
        # self.transforms = transforms
        # self.high_voxel = []
        # self.low_voxel = []
        # self.labels = []



        # # 指定ディレクトリ内のすべての .npy ファイルのパスを取得
        # npy_files = []
        # from tqdm import tqdm

        # for user_id in tqdm(range(1, 28), desc="Processing users", ncols=100, position=1):
        #     # for sub_folder in tqdm(['0', '1'],desc="left or right", ncols=100, position=2, leave=False):  # サブフォルダ0と1
        #     for sub_folder in ['0']:  # サブフォルダ0と1
        #         # パスのパターンを生成
        #         path_pattern = f"{data_path}/user{user_id}/{sub_folder}/npy/*.npy"
        #         # このパターンに合うファイルを検索
        #         npy_files.extend(glob.glob(path_pattern))


        # for file_path in tqdm(npy_files, desc="Loading data"):
        #     file_name = os.path.basename(file_path)
        #     elements = file_name.split('_')
        #     label = [float(elements[1])/1920, float(elements[2])/1680]  
        #     data = np.load(file_path)  
        #     self.data_list.append(data)
        #     self.labels.append(label)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # events_input = self.data_list[idx]
        # events_input = events_input.astype(np.float32)
        # label = self.labels[idx]

        # if self.transforms:
        #     sample = self.transforms(sample)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # event_voxel_low = events_to_voxel_all(events_input, 1, 1, 2, 346, 346, device) 
        # event_voxel_high = events_to_voxel_all(events_input, 1, 1, 8, 346, 346, device) 


        input_path_high = self.dir_input + self.title + str(idx).zfill(5) + f"_{150}.npz"
        input_loaded_high = np.load(input_path_high)
        event_voxel_high = np.expand_dims(input_loaded_high['arr_0'], axis=1)

        input_path_low = self.dir_input + self.title + str(idx).zfill(5) + f"_{30}.npz"
        input_loaded_low = np.load(input_path_low)
        event_voxel_low = np.expand_dims(input_loaded_low['arr_0'], axis=1)

        target_path = self.dir_target + self.title + str(idx).zfill(5) + f"_{15}.npz"
        label = np.load(target_path)['arr_0']
        assert label.shape == (15, 2), f"the shape is not what you expected: {label.shape}"
        assert event_voxel_high.shape == (150, 1, 480, 640), f"the shape is not what you expected: {event_voxel_high.shape}"
        assert event_voxel_low.shape == (30, 1, 480, 640), f"the shape is not what you expected: {event_voxel_low.shape}"
        # print("before normalization: ", label)
        label[:, 0] = (label[:, 0] + 1100) / 240
        label[:, 1] = (label[:, 1] - 650) / 300
         
        # if self.phase == 'train':
        #     event_voxel_low, event_voxel_high = RandomCrop(event_voxel_low, event_voxel_high, (320, 320))
        #     event_voxel_low, event_voxel_high = HorizontalFlip(event_voxel_low, event_voxel_high)
        # else:
        #     event_voxel_low, event_voxel_high  = CenterCrop(event_voxel_low, event_voxel_high, (320, 320))

        # event_voxel_low, event_voxel_high = RandomCrop(event_voxel_low, event_voxel_high, (480, 640))
        # event_voxel_low, event_voxel_high = HorizontalFlip(event_voxel_low, event_voxel_high)

        result = {}
        result['event_low'] = torch.FloatTensor(event_voxel_low)
        result['event_high'] = torch.FloatTensor(event_voxel_high)
        result['label'] = torch.FloatTensor(label)
        return result