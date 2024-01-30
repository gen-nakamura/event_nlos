import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from model import MSTP  
from dataset import CustomDataset 
from earlystopping import EarlyStopping
from torch.utils.data import DataLoader, Subset
import wandb
import time
import os
import numpy as np
from utils.utils import plot_coordinates
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

model_path = "./model/model.pth"


def train(model, train_loader, val_loader, optimizer, loss_function, device, num_epochs, project_name):
    early_stopping = EarlyStopping(patience=7, delta=0.001, model_path=model_path)
    wandb.init(project="lip_reading", name=project_name)
    wandb.watch(loss_function, log="all")

    for epoch in tqdm(range(num_epochs), desc="Epochs", ncols=100, position=0):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Batches", ncols=100, position=1, leave=False):
            # ここでfor文回す
            event_low = batch['event_low'].to(device)
            event_high = batch['event_high'].to(device)
            labels = batch['label'].to(device)

            assert len(labels) == 30, f"label length is not what you expected: {len(labels)}"
            assert len(event_low) == 30, f"event_low length is not what you expected: {len(event_low)}"
            assert len(event_high) == 150, f"event_high length is not what you expected: {len(event_high)}"
            for i in range(30):
                low_batch = event_low[i]
                high_batch = event_high[i*10:(i+1)*10, :]
                label = labels[i]
                outputs = model(low_batch, high_batch)
                assert label.shape == (2,), "label shape is not what you expected"
                loss = loss_function(outputs, label)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

        avg_loss = total_loss / len(train_loader) / 30
        # print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
        wandb.log({"train_loss": avg_loss})
        

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():

            for val_batch in tqdm(val_loader,desc=f"Valodation {epoch+1} Batches", ncols=100, position=2, leave=False): 
                event_low_val = val_batch['event_low'].to(device)
                event_high_val = val_batch['event_high'].to(device)
                labels_val = val_batch['label'].to(device)
                for i in range(30):
                    low_batch = event_low_val[i]
                    high_batch = event_high_val[i*10:(i+1)*10, :]
                    label = labels[i]
                    outputs = model(low_batch, high_batch)
                    assert label.shape == (2,), "label shape is not what you expected"
                    loss = loss_function(outputs, label)

                    outputs_val = model(event_low_val, event_high_val)
                    val_loss += loss_function(outputs_val, labels_val).item()

        avg_val_loss = val_loss / len(val_loader) / 30
        wandb.log({"val_loss": avg_val_loss})
        # print(f'Validation Loss: {avg_val_loss:.4f}')

        # Call EarlyStopping
        early_stopping = EarlyStopping(patience=7, delta=0.001, model_path=model_path)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    wandb.finish()

def test(model, test_loader, test_dataset,loss_function, device):
    model.eval()  # モデルを評価モードに設定
    total_loss = 0
    with torch.no_grad():  # 勾配計算を無効化
        start_time = time.time()  # Start the timer for the entire loop
        loop_count = 0  # Initialize loop counter

        for batch_num, batch in enumerate(tqdm(test_loader, desc="Testing Batches", ncols=100, position=3, leave=False), start=1):

            event_low = batch['event_low'].to(device)
            event_high = batch['event_high'].to(device)
            labels = batch['label'].to(device)
            pred_plot = []

            for i in range(30):
                low_batch = event_low[i]
                high_batch = event_high[i*10:(i+1)*10, :]
                label = labels[i]
                outputs = model(low_batch, high_batch)
                assert label.shape == (2,), "label shape is not what you expected"
                loss = loss_function(outputs, label)

                # モデルによる予測
                outputs = model(event_low, event_high)

                # 損失の計算
                loss = loss_function(outputs, labels)
                total_loss += loss.item()

            if batch_num == 1:
                plot_coordinates(np.array(pred_plot), labels)

    end_time = time.time()
    elapsed_time = end_time - start_time  # Calculate the elapsed time for the entire loop
    avg_time_per_loop = elapsed_time / len(test_dataset) # Calculate the average time per loop
    print(f"Time taken for the entire loop: {elapsed_time:.4f} seconds")
    print(f"Average time taken for one loop iteration: {avg_time_per_loop:.4f} seconds")
    avg_loss = total_loss / len(test_loader) / 30  # 平均損失の計算
    print(f'Test Loss: {avg_loss:.4f}')


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Train the MSTP model')
    # parser.add_argument('--train_data_path', type=str, required=True, help='Path to the training data')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--se', type=bool, default=False)
    parser.add_argument('--base_channel', type=int, default=64)
    parser.add_argument('--alpha', type=int, default=8)
    parser.add_argument('--beta', type=int, default=5)
    parser.add_argument('--t2s_mul', type=int, default=2)
    parser.add_argument('--wandb', type=str,  help='the name of wandb project')
    args = parser.parse_args(args=["--learning_rate", "0.001", "--num_epochs", "70", "--batch_size", "6", "--wandb", "new_model_project_from_lip_reading"])
        

    input_path = '../Dataset/input_data/'
    target_path = '../Dataset/target_data/'
    print("Start loading dataset")
    dataset = CustomDataset(input_path, target_path)

    # Calculate the indices for each subset
    total_size = len(dataset)
    train_size = int(total_size * 0.7)
    val_size = int(total_size * 0.2)
    test_size = total_size - train_size - val_size

    # Split the dataset
    train_dataset = Subset(dataset, range(0, train_size))
    val_dataset = Subset(dataset, range(train_size, train_size + val_size))
    test_dataset = Subset(dataset, range(train_size + val_size, total_size))

    # Create DataLoaders for each set
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cpu") #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MSTP(args).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_function = nn.MSELoss()

    print("Start training")
    train(model, train_loader, val_loader, optimizer, loss_function, device, args.num_epochs, args.wandb)

    print("Training completed")
    torch.save(model.state_dict(), model_path)
    print("Model Saved")

    print("Start Testing")

    test(model, test_loader, test_dataset,loss_function, device)

    end_time = time.time()
    elapsed_time = end_time - start_time 
    print(elapsed_time)

if __name__ == "__main__":
    main()
