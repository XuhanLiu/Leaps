import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import util
from models import generator
from models import classifier
from tqdm import tqdm


def main():
    tgts = pd.read_table("data/kinase_tgt.txt").set_index('accession').sort_index()
    voc_tgt = util.VocTgt(max_len=1000)
    tgt_set = util.TgtData(tgts.sequence.values, tgts.index.values, voc_tgt)
    tgt_loader = DataLoader(tgt_set, batch_size=BATCH_SIZE, collate_fn=tgt_set.collate_fn)
    voc_cmp = util.VocCmp(init_from_file='data/kinase_voc.txt', max_len=100)
    agent_path = 'output/netA_48'
    agent = nn.DataParallel(generator.Seq2Seq(voc_tgt, voc_cmp).to(util.dev), device_ids=util.devices)
    params = torch.load(agent_path + '.pkg')

    agent.load_state_dict(params)
    clf = classifier.MTClassifier(4096, len(tgts))
    clf.load_state_dict(torch.load('output/MT_DNN_kinase_4.pkg'))
    env = util.Activity(clf)

    results = pd.DataFrame()
    for i in tqdm(range(1000)):
        for (ix, tgt) in tgt_loader:
            df = pd.DataFrame()
            df["accession"] = tgts.index.values[ix]
            ix, tgt = ix.to(util.dev).unsqueeze(1), tgt.to(util.dev)
            seqs = agent(tgt)

            df["canonical_smiles"], df['valid'] = util.check_smiles(seqs, voc_cmp)
            df["score"] = env(df.canonical_smiles).gather(1, ix).detach().cpu().numpy()
            results = results.append(df)
    results.to_csv(agent_path + '_mol.txt', index=None, sep='\t')


if __name__ == '__main__':
    BATCH_SIZE = 6
    main()