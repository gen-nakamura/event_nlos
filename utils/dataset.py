from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob

class CustomImageDataset(Dataset):
    def __init__(self, input_dir, target_dir, title, num_bin):
        self.dir_input = input_dir
        self.dir_target = target_dir
        self.title = title
        self.num_bin = num_bin
        # print('custom image dataset')

    def __len__(self):
        return len(glob.glob(self.dir_target + f"*_{self.num_bin}.npz"))

    def __getitem__(self, idx):
        #rd_lr = random.random()
        #rd_ud = random.random()
        # input_path = self.dir_input + self.title + str(idx).zfill(5) + f"_{self.num_bin}.npz"
        input_path = self.dir_input + self.title + "00001" + f"_{self.num_bin}.npz"
        input_loaded = np.load(input_path)
        input = input_loaded['arr_0']
        # target_path = self.dir_target + self.title + str(idx).zfill(5) + f"_{self.num_bin}.npz"
        target_path = self.dir_target + self.title + "00001" + f"_{self.num_bin}.npz"
        target = np.load(target_path)['arr_0']
        assert target.shape == (150, 2), f"the shape is not what you expected: {target.shape}"
        # print("before normalization: ", target)
        
        target[:, 0] = (target[:, 0]  * -1 -860) / 240
        target[:, 1] = (target[:, 1] - 650) / 300
        # print()
        # print("after normalization: ", target)
        # print("\rData No." + str(idx).zfill(5) + " was loaded", end="")
        """
        if(rd_lr < flip_lr):
            input = input[:, ::-1, :]
            target = target[:, ::-1]
        if(rd_ud < flip_ud):
            input = input[::-1, :, :]
            target = target[::-1, :, :]
        """
        return input, target