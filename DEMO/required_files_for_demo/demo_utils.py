import torch
import torch.nn as nn
import pickle
from tqdm import tqdm
import pandas as pd
import argparse
from rdkit import Chem
from utils import *
import rdkit
import networkx as nx

import os
from torch_geometric.data import InMemoryDataset
from torch_geometric import data as DATA
import torch
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Batch
import re
from datetime import datetime


def format_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    smiles = rdkit.Chem.MolToSmiles(mol, isomericSmiles=True)

    return smiles

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set] + [x not in allowable_set]

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) + 
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetFormalCharge(), [-1, -2, 1, 2, 0]) +
                    one_of_k_encoding_unk(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2]) +
                    [atom.GetIsAromatic()] +
                    [atom.IsInRing()])

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

def seq_cat(prot):
    seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
    seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}
    max_seq_len = 1000
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]): 
        x[i] = seq_dict[ch]
    return x

def process_latent_a(smile, protein_seq):
    tokenizer_file = f'data/{smile}_tokenizer.pkl'
    tokenizer = Tokenizer(Tokenizer.gen_vocabs(smile))

    smile_graph = {}
    g = smile_to_graph(smile)
    smile_graph[smile] = g

    with open(tokenizer_file, 'wb') as file:
        pickle.dump(tokenizer, file)
    XT = seq_cat(protein_seq)
    name = datetime.now().strftime("%Y%m%d%H%M%S")
    data = TestbedDataset(root='data', dataset=name, xd=np.asarray([smile]), xt=np.asarray([XT]),  smile_graph=smile_graph)
    
    return data

def process_latent(smile, protein_seq, affinity):
    tokenizer_file = f'data/{smile}_tokenizer.pkl'
    tokenizer = Tokenizer(Tokenizer.gen_vocabs(smile))

    smile_graph = {}
    g = smile_to_graph(smile)
    smile_graph[smile] = g

    with open(tokenizer_file, 'wb') as file:
        pickle.dump(tokenizer, file)
    XT = seq_cat(protein_seq)
    Y = float(affinity)
    Y = np.asarray([Y])
    name = datetime.now().strftime("%Y%m%d%H%M%S")
    data = TestbedDataset2(root='data', dataset=name, xd=np.asarray([smile]), xt=np.asarray([XT]), y=Y, smile_graph=smile_graph)
    
    return data



class Tokenizer:
    NUM_RESERVED_TOKENS = 32
    SPECIAL_TOKENS = ('<sos>', '<eos>', '<pad>', '<mask>', '<sep>', '<unk>')
    SPECIAL_TOKENS += tuple([f'<t_{i}>' for i in range(len(SPECIAL_TOKENS), 32)])  # saved for future use

    PATTEN = re.compile(r'\[[^\]]+\]'
                        # only some B|C|N|O|P|S|F|Cl|Br|I atoms can omit square brackets
                        r'|B[r]?|C[l]?|N|O|P|S|F|I'
                        r'|[bcnops]'
                        r'|@@|@'
                        r'|%\d{2}'
                        r'|.')
    
    ATOM_PATTEN = re.compile(r'\[[^\]]+\]'
                             r'|B[r]?|C[l]?|N|O|P|S|F|I'
                             r'|[bcnops]')

    @staticmethod
    def gen_vocabs(smiles_list):
        smiles_set = set(smiles_list)
        vocabs = set()

        for a in tqdm(smiles_set):
            vocabs.update(re.findall(Tokenizer.PATTEN, a))

        return vocabs

    def __init__(self, vocabs):
        special_tokens = list(Tokenizer.SPECIAL_TOKENS)
        vocabs = special_tokens + sorted(set(vocabs) - set(special_tokens), key=lambda x: (len(x), x))
        self.vocabs = vocabs
        self.i2s = {i: s for i, s in enumerate(vocabs)}
        self.s2i = {s: i for i, s in self.i2s.items()}

    def __len__(self):
        return len(self.vocabs)

    def parse(self, smiles, return_atom_idx=False):
        l = []
        if return_atom_idx:
            atom_idx=[]
        for i, s in enumerate(('<sos>', *re.findall(Tokenizer.PATTEN, smiles), '<eos>')):
            if s not in self.s2i:
                a = 3  # 3 for <mask> !!!!!!
            else:
                a = self.s2i[s]
            l.append(a)
            
            if return_atom_idx and re.fullmatch(Tokenizer.ATOM_PATTEN, s) is not None:
                atom_idx.append(i)
        if return_atom_idx:
            return l, atom_idx
        return l

    def get_text(self, predictions):
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.tolist()

        smiles = []
        for p in predictions:
            s = []
            for i in p:
                c = self.i2s[i]
                if c == '<eos>':
                    break
                s.append(c)
            smiles.append(''.join(s))

        return smiles

class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis', 
                 xd=None, xt=None, transform=None,
                 pre_transform=None,smile_graph=None):

        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.data_l = []
        self.pad_token = Tokenizer.SPECIAL_TOKENS.index('<pad>')
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, xd, xt, smile_graph):
        assert (len(xd) == len(xt) and len(xt)), "The three lists must be the same length!"

        data_list = []
        data_len = len(xd)
        for i in range(data_len):
            print('Preparing data in Pytorch Format: {}/{}'.format(i + 1, data_len))
            smiles = xd[i]
            target = xt[i]
            c_size, features, edge_index, edge_feats = smile_graph[smiles]
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                edge_attr=torch.Tensor(edge_feats))
            GCNData.target = torch.LongTensor([target])
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Data preparation Done!. Saving to file.')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        self.data_l = data_list


    def __len__(self):
        return len(self.data_l)

    def __getitem__(self, idx):
        return self.data_l[idx]

class TestbedDataset2(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis', 
                 xd=None, xt=None, y=None, transform=None,
                 pre_transform=None,smile_graph=None):

        super(TestbedDataset2, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.data_l = []
        self.pad_token = Tokenizer.SPECIAL_TOKENS.index('<pad>')
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt, y, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, xd, xt, y, smile_graph):
        assert (len(xd) == len(xt) and len(xt)), "The three lists must be the same length!"

        data_list = []
        data_len = len(xd)
        for i in range(data_len):
            print('Preparing data in Pytorch Format: {}/{}'.format(i + 1, data_len))
            smiles = xd[i]
            target = xt[i]
            labels = y[i]
            c_size, features, edge_index, edge_feats = smile_graph[smiles]
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                edge_attr=torch.Tensor(edge_feats),
                                y=torch.FloatTensor([labels]))
            GCNData.target = torch.LongTensor([target])
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Data preparation Done!. Saving to file.')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        self.data_l = data_list


    def __len__(self):
        return len(self.data_l)

    def __getitem__(self, idx):
        return self.data_l[idx]

#prepare the protein and drug pairs
def collate(data_list):
    batchA = Batch.from_data_list([data for data in data_list])
    return batchA