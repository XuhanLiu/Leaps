import pandas as pd
from rdkit import Chem
from tqdm import tqdm
import util
import numpy as np
import re
import os


def ligd(input, output):
    df = pd.read_table(input)
    voc = util.VocCmp('data/kinase_voc.txt')
    words = set()
    new = pd.DataFrame()
    for i, row in tqdm(df.iterrows(), total=len(df)):
        smile = re.sub('\[\d+', '[', row.canonical_smiles)
        if '.' in smile:
            frags = smile.split('.')
            ix = np.argmax([len(frag) for frag in frags])
            smile = frags[ix]
        if {'C', 'c'}.isdisjoint(smile): continue
        if '%' in smile: continue
        try:
            smile = smile.replace('[NH+]', 'N').replace('[NH2+]', 'N').replace('[NH3+]', 'N')
            smile = Chem.CanonSmiles(smile, 0)
            token = voc.tokenize(smile)
            if {'[As]', '[Au]', '[Hg]', '[Zn]'}.isdisjoint(token) and len(token) <= 100:
                words.update(token)
                row.canonical_smiles = smile
                row['TOKEN'] = ' '.join(token)
        except:
            print(smile)
        new = new.append(row)
    new.to_csv(output, sep='\t', index=None)


def prot(input, output):
    df = pd.read_table(input)
    for i, row in tqdm(df.iterrows(), total=len(df)):
        if len(row.sequence) > 1000:
            df = df.drop(i)
    df.to_csv(output, sep='\t', index=None)


def validate(fname):
    df = pd.read_table(fname).dropna()
    for i, row in tqdm(df.iterrows(), total=len(df)):
        smile = row.canonical_smiles
        token = row.TOKEN
        code = token.replace(' ', '').replace('L', 'Cl').replace('R', 'Br')[:-3]
        assert smile == code


def refine():
    df = pd.read_table(pair_file)[pair].dropna()
    df = df.groupby(pair[0:2]).mean().unstack('accession')
    cmps = pd.read_table('data/kinase_corpus.txt').set_index(pair[0]).dropna()
    cmp_ix = df.index.intersection(cmps.index)
    cmps = cmps.loc[cmp_ix]
    df = df.loc[cmp_ix].stack().unstack('inchi_key')

    tgt_ix = (df >= 6.5).astype(int).sum(axis=1)
    tgts = pd.read_table('data/kinase_tgt.txt').set_index(pair[1]).dropna()
    tgt_ix = tgt_ix[tgt_ix >= 10].index.intersection(tgts.index)
    tgts = tgts.loc[tgt_ix]
    df = df.loc[tgt_ix]

    cmp_ix = df.columns.levels[1]
    cmps = cmps.loc[cmp_ix]
    print(len(tgt_ix), len(cmp_ix))
    df.stack().to_csv('data/kinase_pair.txt', sep='\t')
    cmps.to_csv('data/kinase_cmp.txt', sep='\t')
    tgts.to_csv('data/kinase_tgt.txt', sep='\t')


if __name__ == '__main__':
    # Zinc('zinc/')
    pair = ['inchi_key', 'accession', 'pchembl_value']
    pair_file = 'data/kinase_pair_raw.txt'
    tgt_files = 'data/kinase_tgt_raw.txt'
    cmp_files = 'data/kinase_cmp_raw.txt'

    # ligd('data/kinase_cmp_raw.txt', 'data/kinase_cmp_corpus.txt')
    prot('data/kinase_tgt_raw.txt', 'data/kinase_tgt.txt')
    refine()