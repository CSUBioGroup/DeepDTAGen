import os
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import auc,precision_recall_curve
from torch_geometric import data as DATA
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import re
from typing import List


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

# tokenizer = Tokenizer(Tokenizer.gen_vocabs(smiles))

#pre_filter is a method that is applied to each individual sample of the dataset before it is added to the list of processed data. The purpose of pre_filter is to remove samples from the dataset that do not meet certain criteria. For example, you might use pre_filter to remove samples that have missing data or that do not meet some quality threshold.
#pre_transform, on the other hand, is a method that is applied to each individual sample after it has been processed by collate, but before it is returned by the data loader. The purpose of pre_transform is to transform the individual samples in some way. For example, you might use pre_transform to normalize the features of the graph, to add noise to the graph, or to perform data augmentation.
class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis', 
                 xd=None, xdt=None, xt=None, y=None, transform=None,
                 pre_transform=None,smile_graph=None):

        #root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        self.pad_token = Tokenizer.SPECIAL_TOKENS.index('<pad>')
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xdt, xt, y,smile_graph)
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

    def process(self, xd, xdt, xt, y, smile_graph):
        assert (len(xd) == len(xt) and len(xt) == len(y) == len(xdt)), "The three lists must be the same length!"

        smi = pad_sequence(xdt, batch_first=True, padding_value=self.pad_token)
        data_list = []
        data_len = len(xd)
        for i in range(data_len):
            print('Preparing data in Pytorch Format: {}/{}'.format(i + 1, data_len))
            smiles = xd[i]
            target = xt[i]
            labels = y[i]
            tok_smi = smi[i]
            # print(tok_smi)
            tok_smi = tok_smi.tolist() 
            c_size, features, edge_index, edge_feats = smile_graph[smiles]
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                edge_attr=torch.Tensor(edge_feats),
                                y=torch.FloatTensor([labels]))
            GCNData.target = torch.LongTensor([target])
            GCNData.target_seq = torch.LongTensor([tok_smi])
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Data preparation Done!. Saving to file.')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def logging(msg, FLAGS):
    fpath = os.path.join(FLAGS.log_dir, f"log_{FLAGS.dataset_name}.txt")
    with open(fpath, "a") as fw:
        fw.write("%s\n" % msg)

def save_best_model(mse_loss, model, best_mse, model_path):
    if mse_loss < best_mse:
        best_mse = mse_loss
        torch.save(model.state_dict(), model_path)
        print("Best model saved!")


def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs
def get_cindex(Y, P):
    P = P[:,np.newaxis] - P
    P = np.float32(P==0) * 0.5 + np.float32(P>0)

    Y = Y[:,np.newaxis] - Y
    Y = np.tril(np.float32(Y>0), 0)

    P_sum = np.sum(P*Y)
    Y_sum = np.sum(Y)


    if Y_sum==0:
        return 0
    else:
        return P_sum/Y_sum

def r_squared_error(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean) )

    return mult / float(y_obs_sq * y_pred_sq)


def get_k(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs*y_pred) / float(sum(y_pred*y_pred))

def squared_error_zero(y_obs,y_pred):
    k = get_k(y_obs,y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k*y_pred)) * (y_obs - (k* y_pred)))
    down= sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))

    return 1 - (upp / float(down))

def get_rm2(ys_orig,ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)
    return r2 * (1 - np.sqrt(np.absolute((r2*r2)-(r02*r02))))

def get_aupr(y_true,y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true,y_pred)
    roc_aupr = auc(recall,precision)
    return roc_aupr
