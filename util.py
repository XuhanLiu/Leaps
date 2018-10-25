import torch as T
from torch.utils.data import Dataset
import re
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import os
from sklearn.externals import joblib
from scipy import linalg
from rdkit import rdBase


rdBase.DisableLog('rdApp.error')
AA = 'ARNDCQEGHILKMFPSTWYV'
dev = T.device('cuda')
# devices = list(range(8))
devices = [0, 2]


class VocTgt:
    def __init__(self, max_len=1000):
        self.chars = ['-'] + [r for r in AA]
        self.size = len(self.chars)
        self.max_len = max_len
        self.vocab = dict(zip(self.chars, range(len(self.chars))))

    def encode(self, seq):
        """Takes a list of characters (eg '[NH]') and encodes to array of indices"""
        smiles_matrix = T.zeros(self.max_len)
        for i in range(len(seq)):
            res = seq[i] if seq[i] in self.chars else '-'
            smiles_matrix[i] = self.vocab[res]
        return smiles_matrix


class VocCmp:
    """A class for handling encoding/decoding from SMILES to an array of indices"""
    def __init__(self, init_from_file=None, max_len=100):
        self.chars = ['EOS', 'GO']
        if init_from_file: self.init_from_file(init_from_file)
        self.size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}
        self.max_len = max_len

    def encode(self, char_list):
        """Takes a list of characters (eg '[NH]') and encodes to array of indices"""
        smiles_matrix = T.zeros(len(char_list))
        for i, char in enumerate(char_list):
            smiles_matrix[i] = self.vocab[char]
        return smiles_matrix

    def decode(self, matrix):
        """Takes an array of indices and returns the corresponding SMILES"""
        chars = []
        for i in matrix:
            if i.item() == self.vocab['EOS']: break
            chars.append(self.reversed_vocab[i.item()])
        smiles = "".join(chars)
        smiles = smiles.replace('L', 'Cl').replace('R', 'Br')
        return smiles

    def tokenize(self, smile):
        """Takes a SMILES and return a list of characters/tokens"""
        regex = '(\[[^\[\]]{1,6}\])'
        smile = re.sub('\[\d+', '[', smile)
        smile = smile.replace('Cl', 'L').replace('Br', 'R')
        tokens = []
        for word in re.split(regex, smile):
            if word == '' or word is None: continue
            if word.startswith('['):
                tokens.append(word)
            else:
                for i, char in enumerate(word):
                    tokens.append(char)
        tokens.append('EOS')
        return tokens

    def init_from_file(self, file):
        """Takes a file containing \n separated characters to initialize the vocabulary"""
        with open(file, 'r') as f:
            chars = f.read().split()
            assert len(set(chars)) == len(chars)
            self.chars += chars


class MolData(Dataset):
    """Custom PyTorch Dataset that takes a file containing \n separated SMILES"""
    def __init__(self, df, voc, token=None):
        self.voc = voc
        if isinstance(df, str) and os.path.exists(df):
            df = pd.read_table(df)
        self.smiles = df.CANONICAL_SMILES.values
        self.tokens = []
        if token is None:
            for smile in self.smiles:
                token = self.voc.tokenize(smile)
                if len(token) > self.voc.max_len: continue
                self.tokens.append(token)
        else:
            for sent in df[token].values:
                token = sent.split(' ')
                self.tokens.append(token)

    def __getitem__(self, i):
        encoded = self.voc.encode(self.tokens[i])
        return encoded

    def __len__(self):
        return len(self.tokens)

    @classmethod
    def collate_fn(cls, arr, max_len=100):
        """Function to take a list of encoded sequences and turn them into a batch"""
        # max_length = max([seq.size(0) for seq in arr])
        collated_arr = T.zeros(len(arr), max_len).long()
        for i, seq in enumerate(arr):
            collated_arr[i, :seq.size(0)] = seq
        return collated_arr


class TgtData(Dataset):
    def __init__(self, seqs, ix, voc):
        self.voc = voc
        self.index = ix
        self.map = {idx: i for i, idx in enumerate(self.index)}
        self.prots = seqs

    def __getitem__(self, i):
        prot = self.voc.encode(self.prots[i])
        return self.map[self.index[i]], prot

    def __len__(self):
        return len(self.prots)

    def collate_fn(self, arr, max_tgt=1000):
        collated_ix = T.zeros(len(arr)).long()
        collated_tgt = T.zeros(len(arr), max_tgt).long()
        for i, (ix, tgt) in enumerate(arr):
            collated_ix[i] = ix
            collated_tgt[i] = tgt
        return collated_ix, collated_tgt


class QSARData(Dataset):
    """Custom PyTorch Dataset that takes a file containing \n separated SMILES"""
    def __init__(self, smiles, labels, voc, is_token=False):
        self.voc = voc
        self.labels = labels
        self.smiles = smiles
        self.tokens = []
        for smile in self.smiles:
            token = smile.split(' ') if is_token else self.voc.tokenize(smile)
            if len(token) > self.voc.max_len: continue
            self.tokens.append(token)

    def __getitem__(self, i):
        encoded = self.voc.encode(self.tokens[i])
        return encoded, self.labels[i]

    def __len__(self):
        return len(self.tokens)

    def collate_fn(self, arr, max_len=100):
        """Function to take a list of encoded sequences and turn them into a batch"""
        smiles_arr = T.zeros(len(arr), self.voc.max_len).long()
        labels_arr = T.zeros(len(arr), self.labels.shape[1])
        for i, (smile, label) in enumerate(arr):
            smiles_arr[i, :smile.size(0)] = smile
            labels_arr[i, :] = T.tensor(label)
        return smiles_arr, labels_arr


class PairData(Dataset):
    """Custom PyTorch Dataset that takes a file containing \n separated SMILES"""
    def __init__(self, df, voc_tgt, voc_cmp, token=None):
        self.voc_tgt = voc_tgt
        self.voc_cmp = voc_cmp
        if isinstance(df, str) and os.path.exists(df):
            df = pd.read_table(df)
        self.prots = df.sequence.values
        if token:
            self.smiles = [tokens.split(' ') for tokens in df.token.values]
        else:
            self.smiles = []
            for i, row in df.iterrows():
                token = self.voc_cmp.tokenize(row.CANONICAL_SMILES)
                if len(token) > self.voc_cmp.max_len: continue
                self.smiles.append(token)

    def __getitem__(self, i):
        # mol = self.smiles[i]
        # tokenized = self.voc.tokenize(mol)
        smile = self.voc_cmp.encode(self.smiles[i])
        prot = self.voc_tgt.encode(self.prots[i])
        return prot, smile

    def __len__(self):
        return len(self.smiles)

    @classmethod
    def collate_fn(cls, arr, max_cmp=100, max_tgt=1000):
        """Function to take a list of encoded sequences and turn them into a batch"""
        # max_length = max([seq.size(0) for seq in arr])
        collated_tgt = T.zeros(len(arr), max_tgt).long()
        collated_cmp = T.zeros(len(arr), max_cmp).long()
        for i, (tgt, cmp) in enumerate(arr):
            collated_tgt[i, :tgt.size(0)] = tgt
            collated_cmp[i, :cmp.size(0)] = cmp
        return collated_tgt, collated_cmp


class PCMData(Dataset):
    """Custom PyTorch Dataset that takes a file containing \n separated SMILES"""
    def __init__(self, df, voc_tgt, voc_cmp, token=None):
        self.voc_tgt = voc_tgt
        self.voc_cmp = voc_cmp
        if isinstance(df, str) and os.path.exists(df):
            df = pd.read_table(df)
        self.prots = [self.voc_tgt.tokenize(seq) for seq in df.SEQUENCE.values]
        self.label = T.Tensor((df['PCHEMBL_VALUE'] > 6.5).astype(float))
        if token:
            self.smiles = [tokens.split(' ') for tokens in df.TOKEN.values]
        else:
            self.smiles = []
            for i, row in df.iterrows():
                token = self.voc_cmp.tokenize(row.CANONICAL_SMILES)
                if len(token) > self.voc_cmp.max_len: continue
                self.smiles.append(token)

                token = self.voc_tgt.tokenize(row.SEQUENCE)
                if len(token) > self.voc_tgt.max_len: continue
                self.prots.append(token)

    def __getitem__(self, i):
        smile = self.voc_cmp.encode(self.smiles[i])
        prot = self.voc_tgt.encode(self.prots[i])
        return prot, smile, self.label[i]

    def __len__(self):
        return len(self.smiles)

    @classmethod
    def collate_fn(cls, arr, max_cmp=100, max_tgt=1000):
        """Function to take a list of encoded sequences and turn them into a batch"""
        collated_tgt = T.zeros(len(arr), max_tgt).long()
        collated_cmp = T.zeros(len(arr), max_cmp).long()
        label_arr = T.zeros(len(arr), 1)
        for i, (tgt, cmp, label) in enumerate(arr):
            collated_tgt[i, :tgt.size(0)] = tgt
            collated_cmp[i, :cmp.size(0)] = cmp
            label_arr[i, :] = label
        return collated_tgt, collated_cmp, label_arr


def grad(tensor):
    """Wrapper for torch.autograd.Variable that also accepts
       numpy arrays directly and automatically assigns it to
       the GPU. Be aware in case some operations are better
       left to the CPU."""
    if isinstance(tensor, np.ndarray):
        tensor = T.from_numpy(tensor)
    if isinstance(tensor, list):
        tensor = T.Tensor(tensor)
    tensor.requires_grad = True
    return tensor.to(dev)
    # return cuda(T.autograd.Variable(tensor))


def unique(arr):
    # Finds unique rows in arr and return their indices
    arr = arr.cpu().numpy()
    arr_ = np.ascontiguousarray(arr).view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
    _, idxs = np.unique(arr_, return_index=True)
    return T.LongTensor(np.sort(idxs)).to(dev)


def check_smiles(seqs, voc):
    valids = []
    smiles = []
    for j, seq in enumerate(seqs.cpu()):
        smile = voc.decode(seq)
        valids.append(1 if Chem.MolFromSmiles(smile) else 0)
        smiles.append(smile)
    return smiles, np.array(valids, dtype=np.byte)


class Activity:
    """Scores based on an ECFP classifier for activity."""
    def __init__(self, clf, radius=6, bit_len=4096):
        self.clf = clf
        self.radius = radius
        self.bit_len = bit_len

    def __call__(self, smiles):
        fps = self.ECFP_from_SMILES(smiles)
        data = T.FloatTensor(fps.values).to(dev)
        preds = self.clf(data)
        return preds

    @classmethod
    def ECFP_from_SMILES(cls, smiles, radius=3, bit_len=4096, index=None):
        fps = np.zeros((len(smiles), bit_len))
        for i, smile in enumerate(smiles):
            mol = Chem.MolFromSmiles(smile)
            if mol is None:
                fps[i, :] = [0] * bit_len
            else:
                arr = np.zeros((1,))
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bit_len)
                DataStructs.ConvertToNumpyArray(fp, arr)
                fps[i, :] = arr
        return pd.DataFrame(fps, index=(smiles if index is None else index))

    @classmethod
    def calculate_frechet_distance(cls, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representive data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representive data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)

    @classmethod
    def calc_ffd(cls, smiles1, smiles2):
        fps1 = cls.ECFP_from_SMILES(smiles1)
        mu1 = np.mean(fps1, axis=0)
        sigma1 = np.cov(fps1, rowvar=False)

        fps2 = cls.ECFP_from_SMILES(smiles2)
        mu2 = np.mean(fps2, axis=0)
        sigma2 = np.cov(fps2, rowvar=False)
        ffd = cls.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        return ffd

    @classmethod
    def calc_fid(cls, model, smiles1, smiles2):
        fps1 = cls.ECFP_from_SMILES(smiles1)
        act1 = model.inception(fps1)
        mu1 = np.mean(act1, axis=0)
        sigma1 = np.cov(act1, rowvar=False)

        fps2 = cls.ECFP_from_SMILES(smiles2)
        act2 = model.inception(fps2)
        mu2 = np.mean(act2, axis=0)
        sigma2 = np.cov(act2, rowvar=False)
        fid = cls.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        return fid

