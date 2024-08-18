import pandas as pd
import numpy as np
import os
import json,pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils import *
import re
from typing import List

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set] + [x not in allowable_set]
    
def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) + #Atom symbol
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + #Number of adjacent atoms
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + # Number of adjacent hydrogens
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + #Implicit valence
                    one_of_k_encoding_unk(atom.GetFormalCharge(), [-1, -2, 1, 2, 0]) + #Formal charge
                    one_of_k_encoding_unk(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2]) + #Hybridization
                    [atom.GetIsAromatic()] + #Aromaticity
                    [atom.IsInRing()] #In ring
                    )

def bond_features(bond):
    bt = bond.GetBondType()
    bond_feats = [0, 0, 0, 0, bond.GetBondTypeAsDouble()]
    if bt == Chem.rdchem.BondType.SINGLE:
        bond_feats = [1, 0, 0, 0, bond.GetBondTypeAsDouble()]
    elif bt == Chem.rdchem.BondType.DOUBLE:
        bond_feats = [0, 1, 0, 0, bond.GetBondTypeAsDouble()]
    elif bt == Chem.rdchem.BondType.TRIPLE:
        bond_feats = [0, 0, 1, 0, bond.GetBondTypeAsDouble()]
    elif bt == Chem.rdchem.BondType.AROMATIC:
        bond_feats = [0, 0, 0, 1, bond.GetBondTypeAsDouble()]
    return np.array(bond_feats)

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    
    c_size = mol.GetNumAtoms()
    
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edge_feats = bond_features(bond)
        edges.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), {'edge_feats': edge_feats}))
        
    g = nx.Graph()
    g.add_edges_from(edges)
    g = g.to_directed()
    edge_index = []
    edge_feats = []
    for e1, e2, feats in g.edges(data=True):
        edge_index.append([e1, e2])
        edge_feats.append(feats['edge_feats'])
        
    return c_size, features, edge_index, edge_feats

def smile_parse(smiles, tokenizer: Tokenizer):
    tokenizer = Tokenizer(Tokenizer.gen_vocabs(smiles))
    smi = tokenizer.parse(smiles)
    return smi

def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]): 
        x[i] = seq_dict[ch]
    return x  
    
seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000

compound_iso_smiles = []
for dt_name in ['kiba','davis', 'bindingdb']:
    opts = ['train','test']
    for opt in opts:
        df = pd.read_csv('data/' + dt_name + '_' + opt + '.csv')
        compound_iso_smiles += list( df['compound_iso_smiles'] )
compound_iso_smiles = set(compound_iso_smiles)
smile_graph = {}
for smile in compound_iso_smiles:
    g = smile_to_graph(smile)
    smile_graph[smile] = g
dir = 'data'
datasets = ['davis', 'kiba', 'bindingdb']
# convert to PyTorch data format
for dataset in datasets:
    processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    tokenizer_file = f'{dir}/{dataset}_tokenizer.pkl'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        df_train = pd.read_csv('data/' + dataset + '_train.csv')
        df_test = pd.read_csv('data/' + dataset + '_test.csv')

        all_smiles = set(df_train['compound_iso_smiles']).union(set(df_test['compound_iso_smiles']))
        tokenizer = Tokenizer(Tokenizer.gen_vocabs(all_smiles))

        with open(tokenizer_file, 'wb') as file:
            pickle.dump(tokenizer, file)        
        # Process train set
        train_drugs, train_prots, train_Y = list(df_train['compound_iso_smiles']), list(df_train['target_sequence']), list(df_train['affinity'])
        XT = [seq_cat(t) for t in train_prots]
        train_drugs, train_prots, train_Y = np.asarray(train_drugs), np.asarray(XT), np.asarray(train_Y)
        train_XD = [torch.LongTensor(tokenizer.parse(smile)) for smile in train_drugs]

        # Process test set
        test_drugs, test_prots, test_Y = list(df_test['compound_iso_smiles']), list(df_test['target_sequence']), list(df_test['affinity'])
        XT = [seq_cat(t) for t in test_prots]
        test_drugs, test_prots, test_Y = np.asarray(test_drugs), np.asarray(XT), np.asarray(test_Y)
        test_XD = [torch.LongTensor(tokenizer.parse(smile)) for smile in test_drugs]

        print('preparing ', dataset + '_train.pt in pytorch format!')
        train_data = TestbedDataset(root='data', dataset=dataset+'_train', xd=train_drugs, xdt=train_XD, xt=train_prots, y=train_Y,smile_graph=smile_graph)
        print('preparing ', dataset + '_test.pt in pytorch format!')
        test_data = TestbedDataset(root='data', dataset=dataset+'_test', xd=test_drugs, xdt=test_XD, xt=test_prots, y=test_Y,smile_graph=smile_graph)
        print(processed_data_file_train, ' and ', processed_data_file_test, ' have been created')        
    else:
        print(processed_data_file_train, ' and ', processed_data_file_test, ' are already created')