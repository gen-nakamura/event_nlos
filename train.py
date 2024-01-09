import torch
import torch.nn as nn
import torchvision.transforms as T
import wandb
from model import E2VIDRecurrent, E2VID
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import glob

input_path = '/home/kentahorikawa/focal_stack/data/processed/synth_rand/Dataset/input_5_norm/'
target_path = '/home/kentahorikawa/focal_stack/data/processed/synth_rand/Dataset/target_5_norm/'
config = "/home/kentahorikawa/E2VID/rpg_e2vid/model/config.json"
SIZE_X = 256
SIZE_Y = 256
learning_rate = 0.0001
epochs = 200
Loss = "MSE"
activation = "ReLU"
num_bin = 5
batch_size=16
norm = "BN"
flip_lr = 0.4
flip_ud = 0.4

now = datetime.now()
#log_path = "./log" + now.strftime("%Y%m%d-%H%M%S") + "/"
#writer = SummaryWriter(log_path)
plt.figure(figsize=(5, 5))
plt.title("Loss")
train_loss = []
test_loss = []

#device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class CustomImageDataset(Dataset):
    def __init__(self, input_dir, target_dir):
        self.dir_input = input_dir
        self.dir_target = target_dir

    def __len__(self):
        return len(glob.glob(self.dir_target + "*"))

    def __getitem__(self, idx):
        #rd_lr = random.random()
        #rd_ud = random.random()
        input_path = self.dir_input + str(idx).zfill(4) + "rosbag.npz"
        input_loaded = np.load(input_path)
        input = input_loaded['arr_0']
        target_path = self.dir_target + str(idx).zfill(4) + "idepth.npy"
        target = np.load(target_path)
        print("\rData No." + str(idx).zfill(4) + " was loaded", end="")
        """
        if(rd_lr < flip_lr):
            input = input[:, ::-1, :]
            target = target[:, ::-1]
        if(rd_ud < flip_ud):
            input = input[::-1, :, :]
            target = target[::-1, :, :]
        """
        return input, target

"""
transform = T.Compose(
    [T.RandomHorizontalFlip(p=0.5),
     T.RandomVerticalFlip(p=0.5),
     T.RandomRotation(degrees=(-20, 20))]
)
"""

dataset = CustomImageDataset(input_dir=input_path, target_dir=target_path)
print(len(dataset))
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - 50, 50])
#print(len(train_dataset))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
print(train_dataloader)
print(test_dataloader)

data_num = len(train_dataset)
parameters = "aug_bin" + str(num_bin) \
    + "_batch" + str(batch_size) \
        + "_normdata" + str(data_num)  \
            + "_lr" + str(learning_rate).split('.')[1] \
                + "_" + activation \
                    + "_" + Loss \
                        + "_epoch"+ str(epochs) \
                            + "_Norm" + str(norm)
save_model = "/home/kentahorikawa/E2VID/rpg_e2vid/pretrained/model_" + parameters + ".pth"

wandb.init(
    project = "event-based_DepthEstimation",

    name = parameters,

    config={
        "train_data_num": data_num,
        "learning_rate": learning_rate,
        "activation": activation,
        "train_loss": Loss,
        "epochs": epochs,
        "num_bin": num_bin,
        "batches": batch_size,
        "Norm": norm
    }
)

def train(dataloader, model, loss_fn, optimizer, epoch_num, batch):
    model.train()
    # X: input, y: target
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        #loss = torch.sum(loss)
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    loss_num = loss.item()
    #writer.add_scalar("Train_Loss", loss_num, epoch_num)
    wandb.log({"train_loss": loss_num})
    train_loss.append(loss_num)


def test(dataloader, model, loss_fn, epoch_num):
    model.eval()
    with torch.no_grad():
        i=0
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            i += 1
    
    loss_num = loss.item()
    #writer.add_scalar("Test_Loss", loss_num, epoch_num)
    wandb.log({"test_loss": loss_num})
    test_loss.append(loss_num)

def main():
    start = datetime.now()

    # Define model
    model = E2VID(config).to(device)
    
    #train_loss_fn = lpips.LPIPS(net='vgg').cuda().forward
    train_loss_fn = nn.MSELoss()
    test_loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"[Epoch {t+1}]")
        train(train_dataloader, model, train_loss_fn, optimizer, t, batch_size)
        test(test_dataloader, model, test_loss_fn, t)
        print("\n-------------------------------")
    end = datetime.now()
    print("Done!")
    torch.save(model, save_model)
    print("Saved PyTorch Model: " + save_model)
    print("Start: " + start.strftime("%Y%m%d-%H%M%S"))
    print("End: " + end.strftime("%Y%m%d-%H%M%S"))

    #writer.close()
    wandb.finish()
    plt.plot(train_loss, label='train')
    plt.plot(test_loss, label='test')
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    main()
