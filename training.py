import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils import *
from model import DeepDTAGen
from FetterGrad import FetterGrad

from tqdm import tqdm
import sys, os
import time
import pickle
import random

seed = 4221
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

if torch.cuda.is_available():
  generator = torch.Generator('cuda').manual_seed(seed)
else:
  generator = torch.Generator().manual_seed(seed)


"""Train the GraphVAE model using the specified data and hyperparameters."""

def train(model, device, train_loader, optimizer, mse_f, epoch, train_data, FLAGS):
    model.train()

    with tqdm(train_loader, desc=f"Epoch {epoch + 1}") as t:
        for i, data in enumerate(t):
            optimizer.zero_grad()
            batch = data.batch.to(device)
            Pridection, new_drug, lm_loss, kl_loss = model(data.to(device))

            mse_loss = mse_f(Pridection, data.y.view(-1, 1).float().to(device))

            train_ci = get_cindex(Pridection.cpu().detach().numpy(), data.y.view(-1, 1).float().cpu().detach().numpy())

            loss = kl_loss * 0.001 + mse_loss + lm_loss
            # loss.backward()
            # optimizer.step()

            losses = [loss, mse_loss] 
            optimizer.ft_backward(losses)
            optimizer.step()
            t.set_postfix(MSE=mse_loss.item(), Train_cindex=train_ci, KL=kl_loss.item(), LM=lm_loss.item())
        msg = f"Epoch {epoch+1}, total loss={loss.item()}, MSE={mse_loss.item()}, KL_loss={kl_loss.item()}, LM={lm_loss.item()}"
        logging(msg, FLAGS)
    return model

def test(model, device, test_loader, dataset, FLAGS):
    """Test the GraphVAE model on the specified data and report the results.""" 
    print('Testing on {} samples...'.format(len(test_loader.dataset)))
    model.eval()
    total_true = torch.Tensor()
    total_predict = torch.Tensor()
    total_loss = 0 

    if dataset == "kiba":
        thresholds = [10.0, 10.50, 11.0, 11.50, 12.0, 12.50]
    else:
        thresholds = [5.0, 5.50, 6.0, 6.50, 7.0, 7.50, 8.0, 8.50]  

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            
            Pridection, new_drug, lm_loss, kl_loss = model(data.to(device))

            total_true = torch.cat((total_true, data.y.view(-1, 1).cpu()), 0)
            total_predict = torch.cat((total_predict, Pridection.cpu()), 0)   
            G = total_true.numpy().flatten()
            P = total_predict.numpy().flatten()
            mse_loss = mse(G, P)
            test_ci = get_cindex(G, P)      
            rm2 = get_rm2(G, P)   
            auc_values = []
            for t in thresholds:
                auc = get_aupr(np.int32(G > t), P)
                auc_values.append(auc) 
            loss = lm_loss + kl_loss
            # total_loss += loss.item() * data.num_graphs
            total_loss += loss.item() * data.num_graphs
            # msg = f"Test Batch: loss={loss.sum().item()}, MSE={mse_loss.item()}, Test c-index={test_ci}, AUCs={auc_values}"
            # logging(msg, FLAGS) # Log the test results for this batch
    return total_loss, lm_loss, kl_loss, mse_loss, test_ci, rm2, auc_values, G, P

def experiment(FLAGS, dataset, device):
    logging('Starting program', FLAGS)

    # Hyperparameters
    BATCH_SIZE = 32
    LR = 0.0002
    NUM_EPOCHS = 500

    # Print hyperparameters
    print(f"Dataset: {dataset}")
    print(f"Device: {device}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LR}")
    print(f"Epochs: {NUM_EPOCHS}")

    # Log hyperparameters
    msg = f"Dataset {dataset}, Device {device}, batch size {BATCH_SIZE}, learning rate {LR}, epochs {NUM_EPOCHS}"
    logging(msg, FLAGS)

    # Load tokenizer
    with open(f'data/{dataset}_tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    # Load processed data
    processed_data_file_train = f"data/processed/{dataset}_train.pt"
    processed_data_file_test = f"data/processed/{dataset}_test.pt"
    if not (os.path.isfile(processed_data_file_train) and os.path.isfile(processed_data_file_test)):
        print("Please run create_data.py to prepare data in PyTorch format!")
    else:
        train_data = TestbedDataset(root="data", dataset=f"{dataset}_train")
        test_data = TestbedDataset(root="data", dataset=f"{dataset}_test")

        # Prepare PyTorch mini-batches
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
        
        # Initialize model, optimizer, and loss function
        model = DeepDTAGen(tokenizer).to(device)
        optimizer = FetterGrad(optim.Adam(model.parameters(), lr=LR))
        mse_f = nn.MSELoss()

        # Train model
        best_mse = float('inf')  
        for epoch in range(NUM_EPOCHS):
            model = train(model, device, train_loader, optimizer, mse_f, epoch, train_data, FLAGS)

            if (epoch + 1) % 20 == 0:
                # Test model
                total_loss, lm_loss, kl_loss, mse_loss, test_ci, rm2, auc_values, G, P = test(model, device, test_loader, dataset, FLAGS)
                filename = f"saved_models/deepdtagen_model_{dataset}.pth"
                if mse_loss < best_mse:
                    best_mse = mse_loss
                    torch.save(model.state_dict(), filename)
                    print('model saved')

                print(f"MSE: {mse_loss.item():.4f}")
                print(f"CI: {test_ci:.4f}")
                print(f"RM2: {rm2:.4f}")
                print(f"LM: {lm_loss.item():.4f}")
                print(f"KL: {kl_loss.item():.4f}")
                print(f"AUCs: {', '.join([f'{auc:.4f}' for auc in auc_values])}")

        # Save estimated and true labels
        folder_path = "Affinities/"
        np.savetxt(folder_path + f"estimated_labels_{dataset}.txt", P)
        np.savetxt(folder_path + f"true_labels_{dataset}.txt", G)

        logging('Program finished', FLAGS)

if __name__ == "__main__":
    # Dataset names and device setup
    datasets = ['davis', 'kiba', 'bindingdb']
    dataset_idx = int(sys.argv[1])
    dataset = datasets[dataset_idx]
    device = torch.device("cuda:" + str(int(sys.argv[2])) if len(sys.argv) > 2 and torch.cuda.is_available() else "cpu")

    # Flags setup
    FLAGS = lambda: None
    FLAGS.log_dir = 'logs'
    FLAGS.dataset_name = f'dataset_{dataset}'.format(int(time.time()))

    # Create necessary directories
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    if not os.path.exists('Affinities'):
        os.mkdir('Affinities')
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')

    # Run experiment
    experiment(FLAGS, dataset, device)
