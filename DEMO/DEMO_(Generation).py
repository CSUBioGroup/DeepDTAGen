import torch
import torch.nn as nn
import pickle
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader
from required_files_for_demo.demo_utils import *

from required_files_for_demo.model_gen import DeepDTAGen

def demo():
    dataset_name = 'bindingdb'

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths
    model_path = f'./models/deepdtagen_model_{dataset_name}.pth'
    tokenizer_path = f'./data/{dataset_name}_tokenizer.pkl'

    smiles = "O=C(c1nc(NS(=O)(=O)c2cc(Br)cc(Cl)c2O)cn1C1CCCC1)N1CCC(C2CCCN2)CC1"
    protein_sequence = "MATEEKKPETEAARAQPTPSSSATQSKPTPVKPNYALKFTLAGHTKAVSSVKFSPNGEWLASSSADKLIKIWGAYDGKFEKTISGHKLGISDVAWSSDSNLLVSASDDKTLKIWDVSSGKCLKTLKGHSNYVFCCNFNPQSNLIVSGSFDESVRIWDVKTGKCLKTLPAHSDPVSAVHFNRDGSLIVSSSYDGLCRIWDTASGQCLKTLIDDDNPPVSFVKFSPNGKYILAATLDNTLKLWDYSKGKCLKTYTGHKNEKYCIFANFSVTGGKWIVSGSEDNLVYIWNLQTKEIVQKLQGHTDVVISTACHPTENIIASAALENDKTIKLWKSDC"
    conditional_affinity = 5.0
    # Load tokenizer
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    # Load model
    model = DeepDTAGen(tokenizer)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Load test data
    processed_data = f'./data/processed/{smiles}.pt'
    if not os.path.isfile(processed_data):
        test_data = process_latent(smiles, protein_sequence, conditional_affinity)
    else:
        test_data = torch.load(processed_data)
    print(test_data)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=collate)

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        for data in tqdm(test_loader, desc='Testing'):
            res = tokenizer.get_text(model.generate(data.to(device)))
        print("Generated Drug :", res)

if __name__ == "__main__":
    demo()
