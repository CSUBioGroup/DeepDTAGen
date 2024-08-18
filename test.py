import torch
import torch.nn as nn
import pickle
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader
import argparse

from utils import *
from model import DeepDTAGen
from datasets import TestbedDataset

def main(dataset_name):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths
    model_path = f'models/deepdtagen_model_{dataset_name}.pth'
    tokenizer_path = f'data/{dataset_name}_tokenizer.pkl'
    test_batch_size = 128

    # Threshold values based on the dataset
    if dataset_name == 'kiba':
        thresholds = [10.0, 10.50, 11.0, 11.50, 12.0, 12.50]
    elif dataset_name == 'davis':
        thresholds = [5.0, 5.50, 6.0, 6.50, 7.0, 7.50, 8.0, 8.50]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Load tokenizer
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    # Load model
    model = DeepDTAGen(tokenizer)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Load test data
    test_data = TestbedDataset(root='data', dataset=f'{dataset_name}_test')
    test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

    # Evaluate the model
    model.eval()
    total_predict = torch.Tensor().to(device)
    total_true = torch.Tensor().to(device)

    with torch.no_grad():
        for data in tqdm(test_loader, desc='Testing'):
            predictions, _, lm_loss, kl_loss = model(data.to(device))
            total_true = torch.cat((total_true, data.y.view(-1, 1)), dim=0)
            total_predict = torch.cat((total_predict, predictions), dim=0)

        # Convert to numpy arrays
        ground_truth = total_true.cpu().numpy().flatten()
        predicted = total_predict.cpu().numpy().flatten()

        # Calculate metrics
        mse_loss = mse(ground_truth, predicted)
        concordance_index = get_cindex(ground_truth, predicted)
        rm2_value = get_rm2(ground_truth, predicted)
        rms_error = rmse(ground_truth, predicted)
        pearson_corr = pearson(ground_truth, predicted)
        spearman_corr = spearman(ground_truth, predicted)

        # Calculate AUC values for each threshold
        auc_values = [
            get_aupr((predictions.cpu() > threshold).int(), data.y.view(-1, 1).float().cpu())
            for threshold in thresholds
        ]

        # Print the results
        print(f'MSE: {mse_loss:.4f}, CI: {concordance_index:.4f}, RM2: {rm2_value:.4f}')
        print(f'RMS Error: {rms_error}')
        print(f'PPC: {pearson_corr:.4f}, Spearman: {spearman_corr:.4f}')
        print(f'AUC Values: {auc_values}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate DeepDTAGen on a dataset.')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset (e.g., kiba, davis, bindingdb)')
    args = parser.parse_args()

    main(args.dataset)