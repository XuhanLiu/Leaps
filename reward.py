import os
from models import classifier
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, KFold
from torch.utils.data import DataLoader, TensorDataset
import util


def DNN(X, y, X_ind, y_ind, out, reg=False):
    if 'MT_' in out or reg:
        folds = KFold(5).split(X)
        NET = classifier.MTRegressor if reg else classifier.MTClassifier
    else:
        folds = StratifiedKFold(5).split(X, y[:, 0])
        NET = classifier.STRegressor if reg else classifier.STClassifier
    indep_set = TensorDataset(torch.Tensor(X_ind), torch.Tensor(y_ind))
    indep_loader = DataLoader(indep_set, batch_size=BATCH_SIZE)
    cvs = np.zeros(y.shape)
    inds = np.zeros(y_ind.shape)
    for i, (trained, valided) in enumerate(folds):
        train_set = TensorDataset(torch.Tensor(X[trained]), torch.Tensor(y[trained]))
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE)
        valid_set = TensorDataset(torch.Tensor(X[valided]), torch.Tensor(y[valided]))
        valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE)
        net = NET(X.shape[1], y.shape[1])
        net.fit(train_loader, valid_loader, out='%s_%d' % (out, i), epochs=N_EPOCH, lr=LR)
        cvs[valided] = net.predict(valid_loader)
        inds += net.predict(indep_loader)
    return cvs, inds / 5


def Attn(X, y, X_ind, y_ind, out):
    folds = KFold(5).split(X)
    NET = classifier.AttnClassifier
    voc_cmp = util.VocCmp(init_from_file='data/kinase_voc.txt', max_len=100)
    indep_set = util.QSARData(X_ind, y_ind, voc_cmp, is_token=True)
    indep_loader = DataLoader(indep_set, batch_size=BATCH_SIZE, collate_fn=indep_set.collate_fn)
    cvs = np.zeros(y.shape)
    inds = np.zeros(y_ind.shape)
    for i, (trained, valided) in enumerate(folds):
        train_set = util.QSARData(X, y, voc_cmp, is_token=True)
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, collate_fn=indep_set.collate_fn)
        valid_set = util.QSARData(X, y, voc_cmp, is_token=True)
        valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, collate_fn=indep_set.collate_fn)
        net = NET(voc_cmp, 128, 512, y.shape[1])
        net.fit(train_loader, valid_loader, out='%s_%d' % (out, i), epochs=N_EPOCH, lr=LR)
        cvs[valided] = net.predict(valid_loader)
        inds += net.predict(indep_loader)
    return cvs, inds / 5


def mt_dnn():
    pair = ['inchi_key', 'accession', 'pchembl_value']
    cmps = pd.read_table(cmp_files).set_index(pair[0]).dropna()
    df = pd.read_table(pair_file)[pair].dropna()
    df = (df.groupby(pair[0:2]).mean() >= 0).astype(float).unstack('accession')

    test_y = df.sample(20000)
    data_y = df
    # data_y = df.drop(test_y.index)
    test_x = util.Activity.ECFP_from_SMILES(cmps.loc[test_y.index].canonical_smiles, index=test_y.index)
    data_x = util.Activity.ECFP_from_SMILES(cmps.loc[data_y.index].canonical_smiles, index=data_y.index)

    out = 'output/MT_DNN_kinase'
    data_p, test_p = DNN(data_x.values, data_y.values, test_x.values, test_y.values, out=out)
    data, test = data_y.stack(), test_y.stack()
    data['score'] = pd.DataFrame(data_p, index=data_y.index, columns=data_y.columns).stack()
    test['score'] = pd.DataFrame(test_p, index=test_y.index, columns=test_y.columns).stack()
    data.to_csv(out + '.cv.txt')
    test.to_csv(out + '.ind.txt')


def mt_attn():
    pair = ['inchi_key', 'accession', 'pchembl_value']
    cmps = pd.read_table(cmp_files).set_index(pair[0]).dropna()[['token']]
    df = pd.read_table(pair_file)[pair].dropna()
    df = (df.groupby(pair[0:2]).mean() >= 6.5).astype(float).unstack('accession')

    test_y = df.sample(20000)
    data_y = df
    # data_y = df.drop(test_y.index)
    test_x = cmps.loc[test_y.index].token
    data_x = cmps.loc[data_y.index].token

    out = 'output/MT_ATT_kinase'
    data_p, test_p = Attn(data_x.values, data_y.values, test_x.values, test_y.values, out=out)
    data, test = data_y.stack(), test_y.stack()
    data['score'] = pd.DataFrame(data_p, index=data_y.index, columns=data_y.columns).stack()
    test['score'] = pd.DataFrame(test_p, index=test_y.index, columns=test_y.columns).stack()
    data.to_csv(out + '.cv.txt')
    test.to_csv(out + '.ind.txt')


if __name__ == '__main__':
    pair_file = 'data/kinase_pair.txt'
    tgt_files = 'data/kinase_tgt.txt'
    cmp_files = 'data/kinase_cmp.txt'
    BATCH_SIZE = int(2 ** 12)
    N_EPOCH = 1000
    torch.set_num_threads(1)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    LR = 1e-5

    BATCH_SIZE = 256
    mt_attn()