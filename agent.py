#!/usr/bin/env python
import torch
from rdkit import rdBase
from models import generator
import util
import pandas as pd
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F

rdBase.DisableLog('rdApp.error')
torch.set_num_threads(1)
BATCH_SIZE = 6


def main():
    pair = ['inchi_key', 'accession', 'pchembl_value']
    voc_tgt = util.VocTgt(max_len=1000)
    voc_cmp = util.VocCmp(init_from_file='data/kinase_voc.txt', max_len=100)
    agent_path = 'output/netA_%d' % (BATCH_SIZE)

    agent = generator.EncDec(voc_tgt, voc_cmp).to(util.dev)
    df = pd.read_table("data/kinase_pair.txt")[pair]
    cmps = pd.read_table("data/kinase_cmp.txt")
    cmps = cmps.set_index(pair[0])[['token']]
    tgts = pd.read_table("data/kinase_tgt.txt").set_index(pair[1])[['sequence']]
    df = df.join(cmps, on=pair[0])
    df = df.join(tgts, on=pair[1]).dropna()
    data = util.PairData(df, voc_tgt, voc_cmp, token=True)
    data = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=data.collate_fn)
    agent.fit(data, epochs=1000, out=agent_path)


if __name__ == "__main__":
    main()