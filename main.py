import torch
import torchvision
import torch.nn as nn
import numpy as np
import pandas as pd
import wandb
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
from model.model import UNet_2D
from utils.utils import EarlyStopping, plot_coordinates
from utils.dataset import CustomImageDataset
torch.manual_seed(0)
import sys
import time

# x:-860 to -1100
# y: 650 to 950

# バージョン確認
print("torch       :", torch.__version__) # expect 2.1.2
print("torchvision :", torchvision.__version__) # expect 0.16.2

# GPUの確認
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # expect cuda
print("device      :", device)
print("torch cuda  :", torch.version.cuda) # expect 12.1
torch.cuda.empty_cache()
# print(torch.cuda.memory_summary(device=None, abbreviated=False))


# 変更する必要のある変数
TITLE = "BrandSilence"
num_bin = 150

input_path = './Dataset/input_data/'
target_path = './Dataset/target_data/'
SIZE_X = 640
SIZE_Y = 480
learning_rate = 0.0001
epochs = 20
Loss = "MSE"
activation = "ReLU"
batch_size=6
norm = "BN"
flip_lr = 0.4
flip_ud = 0.4

def train(train_dl, val_dl, model, loss_fn, optimizer, epochs, batch, model_path):
    early_stopping = EarlyStopping(patience=7, delta=0.001, model_path=model_path)
    # sys.stderr.write("error in train")
    for epoch in range(epochs):
        model.train()
        total_loss=0
        with tqdm(enumerate(train_dl), desc="[Epoch %d]" % (epoch), total=len(train_dl), leave=False, position=0) as pbar_loss:
            for batch, (X, y) in pbar_loss:
                # sys.stderr.write("error in train batch")
                X, y = X.to(device).to(torch.float), y.to(device).to(torch.float)
                # modelを変更の余地あり
                pred = model(X)
                loss = loss_fn(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        avg_loss = total_loss / len(train_dl)
        # writer.add_scalar("Train_Loss", avg_loss, epoch_num)
        wandb.log({"train_loss": avg_loss})
        print('last batch value: ', batch)
        print('avg_loss: ', avg_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch, (X, y) in tqdm(enumerate(val_dl), desc=f"Valodation {epoch+1} Batches", ncols=100, position=2, leave=False, total=len(val_dl)):
                X, y = X.to(device).to(torch.float), y.to(device).to(torch.float)
                pred = model(X)
                loss = loss_fn(pred, y)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_dl)
        wandb.log({"val_loss": avg_val_loss})

        # Call EarlyStopping
        early_stopping = EarlyStopping(patience=7, delta=0.001, model_path=model_path)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

def test(test_dl, model, loss_fn):
    # prediction = np.array()
    model.eval()
    total_loss = 0
    start_time = time.time()
    with torch.no_grad():
        for batch, (X, y) in tqdm(enumerate(test_dl), desc="Testing Batches", ncols=100, position=3, leave=False, total=len(test_dl)):
            X, y = X.to(device).to(torch.float), y.to(device).to(torch.float)
            pred = model(X)
            print('prediction shape: ', pred.shape)
            loss = loss_fn(pred, y)

            total_loss += loss.item()
            break
    pred_np = pred.to('cpu').numpy()
    # print('pred_np shape: ', pred_np.shape)
    y_np = y.to('cpu').numpy()
    print(f'y_np shape: {y_np.shape}, pred_np shape: {pred_np.shape}')
    print('this is y_np: ', y_np[0])
    print('this is pred_np: ', pred_np[0])
    for i in range(len(pred)):
        plot_coordinates(pred_np[i], y_np[i])
    end_time = time.time()
    elapsed_time = end_time - start_time  # Calculate the elapsed time for the entire loop
    avg_time_per_loop = elapsed_time / len(test_dl) # Calculate the average time per loop
    print(f"Time taken for the entire loop: {elapsed_time:.4f} seconds")
    print(f"Average time taken for one loop iteration: {avg_time_per_loop:.4f} seconds")
    avg_loss = total_loss / len(test_dl)
    print(f"test_loss: {avg_loss:.4f}")

def main():
    start = time.time()

    dataset = CustomImageDataset(input_dir=input_path, target_dir=target_path, title=TITLE, num_bin=num_bin)
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - 300, 200, 100]) # (700, 200, 100)になるように
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    data_num = len(train_dataset)
    parameters = "aug_bin" + str(num_bin) \
        + "_batch" + str(batch_size) \
            + "_normdata" + str(data_num)  \
                + "_lr" + str(learning_rate).split('.')[1] \
                    + "_" + activation \
                        + "_" + Loss \
                            + "_epoch"+ str(epochs) \
                                + "_Norm" + str(norm)
    model_path = "./Dataset/pretrained/model_" + parameters + ".pth"
    wandb.init(
        project = "event-based_NLOS",
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

    # define model
    unet = UNet_2D().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate)

    print('training start')
    train(train_dataloader, validation_dataloader, unet, loss_fn, optimizer, epochs, batch_size, model_path)
    print('training completed, start testing')
    wandb.finish()
    torch.save(unet.state_dict(), model_path)
    test(test_dataloader, unet, loss_fn)
    end = time.time()
    print("Done!")
    print("Saved PyTorch Model: " + model_path)
    elapsed_time = end - start
    print(f"It took {elapsed_time}s to run the file.")


if __name__ == "__main__":
    main()
